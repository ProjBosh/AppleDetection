// src/main/java/volsu/Trainer.java
package volsu;

import volsu.model.ClassPartition;
import volsu.model.ModelConfig;
import volsu.model.Sample;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Режим {@code --train}: формирование выборки, разбиение 70/30, выбор рабочего словаря
 * признаков (3–5 информативных), подбор решающих границ и сохранение модели.
 * <p>
 * <b>Выбор признаков.</b> Из признаков отбирается пул кандидатов по критерию Фишера
 * {@code F = (μ₁ − μ₀)² / (σ₁² + σ₀²)}, затем выполняется жадный прямой отбор:
 * на каждом шаге добавляется признак, дающий наибольшую сбалансированную точность
 * при 3-кратной стратифицированной кросс-валидации (с подбором границ внутри каждого фолда).
 * <p>
 * <b>Подбор границ.</b> Целевая функция — сбалансированная точность
 * {@code BA = (recall(яблоко) + recall(не яблоко)) / 2} на обучающей выборке: она не зависит от
 * соотношения классов и цен ошибок. Границы строгих интервалов подбираются координатным спуском
 * по кандидатам-квантилям значений <b>обоих</b> классов; затем перебираются параметры правила
 * (число голосов a и b) и относительный запас широких интервалов (по чистоте класса
 * ΩA1_2: доля «не яблок» минус удвоенная доля яблок в нём).
 * <p>
 * <b>Разбиение A2</b> (два класса, без класса сомнения) обучается отдельно: широкие интервалы
 * совпадают со строгими, допускается один выброс (b = 2), а словарь признаков подбирается заново.
 */
public class Trainer {

    /** Доля обучающей выборки. */
    public static final double TRAIN_RATIO = 0.7;
    /** Зерно генератора для воспроизводимого разбиения. */
    public static final long SEED = 42L;
    /** Минимальный размер рабочего словаря признаков. */
    public static final int MIN_FEATURES = 3;
    /** Максимальный размер рабочего словаря признаков. */
    public static final int MAX_FEATURES = 5;
    /** Размер масштабирования по умолчанию. */
    public static final int DEFAULT_SIZE = 256;

    private static final Set<String> IMAGE_EXT = Set.of("jpg", "jpeg", "png", "bmp", "gif");
    /** Признаки, не пригодные для интервальных границ (циклический тон; дублирующее соотношение сторон). */
    private static final Set<String> EXCLUDED = Set.of("meanHue", "aspectRatio");

    private static final int CANDIDATE_POOL = 16;
    private static final int CV_FOLDS = 3;
    private static final int CUT_POINTS = 24;
    private static final int SWEEPS_FINAL = 4;
    private static final int SWEEPS_CV = 2;
    private static final double MIN_GAIN = 1e-4;
    private static final double[] LOOSE_MARGINS = {0.05, 0.1, 0.25, 0.5, 1.0, 2.0};

    /**
     * Параметры обучаемой модели: признаки (индексы в полном векторе), границы и правило голосования.
     */
    private static final class Params {
        final int[] cols;
        final double[] sLo;
        final double[] sHi;
        final double[] lLo;
        final double[] lHi;
        final double[] sd;
        final boolean tight;
        double margin;
        int minStrict;
        int rejectVotes;

        /**
         * Создаёт параметры с правилом по умолчанию (a = k, b = 1).
         *
         * @param cols  индексы признаков
         * @param sLo   нижние границы строгих интервалов
         * @param sHi   верхние границы строгих интервалов
         * @param sd    стандартные отклонения признаков
         * @param tight {@code true} — без класса сомнения (широкие интервалы равны строгим)
         */
        Params(int[] cols, double[] sLo, double[] sHi, double[] sd, boolean tight) {
            this.cols = cols;
            this.sLo = sLo;
            this.sHi = sHi;
            this.sd = sd;
            this.tight = tight;
            this.lLo = new double[cols.length];
            this.lHi = new double[cols.length];
            this.margin = tight ? 0.0 : 0.25;
            this.minStrict = cols.length;
            this.rejectVotes = 1;
            refresh();
        }

        /**
         * Пересчитывает широкие интервалы: {@code l = s ± margin · max(sHi − sLo, 0.1·sd)}.
         */
        void refresh() {
            double m = tight ? 0.0 : margin;
            for (int j = 0; j < cols.length; j++) {
                double w = Math.max(sHi[j] - sLo[j], 0.1 * sd[j]);
                lLo[j] = sLo[j] - m * w;
                lHi[j] = sHi[j] + m * w;
            }
        }
    }

    /** Матрица признаков и метки образцов. */
    private static final class Data {
        final double[][] x;
        final int[] y;

        /**
         * Создаёт набор данных.
         *
         * @param x векторы признаков
         * @param y метки (0 — яблоко, 1 — не яблоко)
         */
        Data(double[][] x, int[] y) {
            this.x = x;
            this.y = y;
        }

        /**
         * Строит набор данных из образцов с извлечёнными признаками.
         *
         * @param samples образцы
         * @return набор данных
         */
        static Data of(List<Sample> samples) {
            List<Sample> ok = new ArrayList<>();
            for (Sample s : samples) {
                if (s.getFeatures() != null) {
                    ok.add(s);
                }
            }
            double[][] x = new double[ok.size()][];
            int[] y = new int[ok.size()];
            for (int i = 0; i < ok.size(); i++) {
                x[i] = ok.get(i).getFeatures();
                y[i] = ok.get(i).getLabel();
            }
            return new Data(x, y);
        }
    }

    /** Фолды стратифицированной кросс-валидации. */
    private static final class Folds {
        final double[][][] trainX = new double[CV_FOLDS][][];
        final int[][] trainY = new int[CV_FOLDS][];
        final double[][][] valX = new double[CV_FOLDS][][];
        final int[][] valY = new int[CV_FOLDS][];
    }

    /**
     * Основной цикл обучения для заданного разбиения.
     *
     * @param dataDir   папка с данными ({@code data/} с подпапками apple и other)
     * @param partition разбиение классов (A1, A2, ...)
     * @return обученная модель
     * @throws IOException           при ошибках чтения данных
     * @throws IllegalStateException если в обучающей выборке нет образцов одного из классов
     */
    public ModelConfig train(File dataDir, ClassPartition partition) throws IOException {
        List<Sample> samples = loadDataset(dataDir);
        List<List<Sample>> split = splitTrainTest(samples, TRAIN_RATIO);
        writeMetadata(dataDir, samples);
        List<Sample> train = extractAll(split.get(0), DEFAULT_SIZE, DEFAULT_SIZE);
        return fitPartition(train, partition);
    }

    /**
     * Обучает модель на уже подготовленной обучающей выборке: выбор признаков, подбор границ,
     * калибровка уверенности класса «похоже на яблоко».
     *
     * @param train     обучающие образцы с признаками
     * @param partition разбиение
     * @return модель
     */
    private ModelConfig fitPartition(List<Sample> train, ClassPartition partition) {
        boolean tight = partition.size() < 3;
        List<String> features = selectFeatures(train, tight);
        ModelConfig cfg = fitBoundaries(train, features, partition);
        return clusterUnsure(cfg, train);
    }

    /**
     * Читает выборку. Если рядом с папкой данных есть {@code python/metadata.csv}
     * (заголовок {@code path,class,split}), из него берутся пути и назначенные подвыборки;
     * изображения из {@code data/apple} и {@code data/other}, которых в файле нет, добавляются
     * без подвыборки (она назначается в {@link #splitTrainTest(List, double)}).
     * Файл другого формата (например, список источников) игнорируется.
     *
     * @param dataDir папка данных
     * @return список образцов
     * @throws IOException если папка отсутствует или выборка пуста
     */
    public static List<Sample> loadDataset(File dataDir) throws IOException {
        if (dataDir == null || !dataDir.isDirectory()) {
            throw new IOException("Папка с данными не найдена: " + dataDir);
        }
        File root = dataDir.getAbsoluteFile().getParentFile();
        File meta = root == null ? null : new File(root, "python/metadata.csv");
        List<Sample> samples = new ArrayList<>();
        Set<Path> known = new HashSet<>();
        if (meta != null && meta.isFile()) {
            List<String> lines = Files.readAllLines(meta.toPath(), StandardCharsets.UTF_8);
            if (!lines.isEmpty() && lines.get(0).trim().toLowerCase(Locale.ROOT).startsWith("path")) {
                for (int i = 1; i < lines.size(); i++) {
                    String line = lines.get(i).trim();
                    if (line.isEmpty()) {
                        continue;
                    }
                    String[] p = line.split(",", -1);
                    if (p.length < 3) {
                        continue;
                    }
                    File f = new File(root, p[0].trim());
                    if (!f.isFile()) {
                        System.err.println("Предупреждение: файл из metadata.csv не найден: " + f);
                        continue;
                    }
                    int label = p[1].trim().equalsIgnoreCase("apple") ? Sample.APPLE : Sample.OTHER;
                    String raw = p[2].trim().toLowerCase(Locale.ROOT);
                    String split = raw.equals("train") || raw.equals("test") ? raw : null;
                    samples.add(new Sample(f, label, split));
                    known.add(f.toPath().toAbsolutePath().normalize());
                }
            }
        }
        List<Sample> scanned = new ArrayList<>();
        scan(new File(dataDir, "apple"), Sample.APPLE, scanned);
        scan(new File(dataDir, "other"), Sample.OTHER, scanned);
        for (Sample s : scanned) {
            if (!known.contains(s.getFile().toPath().toAbsolutePath().normalize())) {
                samples.add(s);
            }
        }
        if (samples.isEmpty()) {
            throw new IOException("В папке " + dataDir + " нет изображений (ожидаются apple/ и other/)");
        }
        return samples;
    }

    /**
     * Добавляет в список изображения из папки (отсортированные по имени).
     *
     * @param dir     папка
     * @param label   метка класса
     * @param samples целевой список
     */
    private static void scan(File dir, int label, List<Sample> samples) {
        File[] files = dir.listFiles();
        if (files == null) {
            return;
        }
        Arrays.sort(files);
        for (File f : files) {
            String n = f.getName().toLowerCase(Locale.ROOT);
            int dot = n.lastIndexOf('.');
            if (f.isFile() && dot >= 0 && IMAGE_EXT.contains(n.substring(dot + 1))) {
                samples.add(new Sample(f, label, null));
            }
        }
    }

    /**
     * Стратифицированное разбиение train/test (по умолчанию 70/30, детерминированное).
     * Образцы с уже назначенной подвыборкой (из metadata.csv) сохраняют её; новые образцы
     * распределяются так, чтобы доля обучающей части в каждом классе была близка к {@code ratio}.
     *
     * @param samples образцы
     * @param ratio   доля обучающей выборки
     * @return список из двух списков: [0] — train, [1] — test
     */
    public static List<List<Sample>> splitTrainTest(List<Sample> samples, double ratio) {
        List<Sample> train = new ArrayList<>();
        List<Sample> test = new ArrayList<>();
        for (int label = 0; label < 2; label++) {
            List<Sample> fresh = new ArrayList<>();
            int trainAssigned = 0;
            int testAssigned = 0;
            for (Sample s : samples) {
                if (s.getLabel() != label) {
                    continue;
                }
                if (s.getSplit() == null) {
                    fresh.add(s);
                } else if (s.getSplit().equals("test")) {
                    test.add(s);
                    testAssigned++;
                } else {
                    train.add(s);
                    trainAssigned++;
                }
            }
            int total = trainAssigned + testAssigned + fresh.size();
            int target = total <= 1 ? total : (int) Math.round(total * ratio);
            if (total > 1) {
                target = Math.max(1, Math.min(total - 1, target));
            }
            int need = Math.max(0, target - trainAssigned);
            Collections.shuffle(fresh, new Random(SEED + label));
            for (int i = 0; i < fresh.size(); i++) {
                Sample s = fresh.get(i);
                if (i < need) {
                    s.setSplit("train");
                    train.add(s);
                } else {
                    s.setSplit("test");
                    test.add(s);
                }
            }
        }
        List<List<Sample>> res = new ArrayList<>();
        res.add(train);
        res.add(test);
        return res;
    }

    /**
     * Записывает {@code python/metadata.csv} (путь, класс, split). Файл другого формата
     * (например, список источников изображений) предварительно сохраняется как
     * {@code python/metadata_sources_backup.csv}.
     *
     * @param dataDir папка данных
     * @param samples образцы с назначенной подвыборкой
     * @throws IOException при ошибке записи
     */
    private static void writeMetadata(File dataDir, List<Sample> samples) throws IOException {
        File root = dataDir.getAbsoluteFile().getParentFile();
        if (root == null) {
            return;
        }
        File meta = new File(root, "python/metadata.csv");
        if (meta.isFile()) {
            List<String> old = Files.readAllLines(meta.toPath(), StandardCharsets.UTF_8);
            boolean ours = !old.isEmpty() && old.get(0).trim().toLowerCase(Locale.ROOT).startsWith("path");
            if (!ours) {
                Files.move(meta.toPath(), new File(root, "python/metadata_sources_backup.csv").toPath(),
                        StandardCopyOption.REPLACE_EXISTING);
            }
        }
        Path rootPath = root.toPath().toAbsolutePath().normalize();
        StringBuilder sb = new StringBuilder("path,class,split\n");
        for (Sample s : samples) {
            if (s.getSplit() == null) {
                continue;
            }
            Path file = s.getFile().toPath().toAbsolutePath().normalize();
            String rel;
            try {
                rel = rootPath.relativize(file).toString().replace('\\', '/');
            } catch (IllegalArgumentException e) {
                rel = file.toString().replace('\\', '/');
            }
            sb.append(rel).append(',').append(s.getLabel() == Sample.APPLE ? "apple" : "other")
                    .append(',').append(s.getSplit()).append('\n');
        }
        Files.createDirectories(meta.toPath().getParent());
        Files.writeString(meta.toPath(), sb.toString(), StandardCharsets.UTF_8);
    }

    /**
     * Извлекает признаки для всех образцов (загрузка → RGB → масштабирование → признаки).
     * Нечитаемые изображения пропускаются с предупреждением.
     *
     * @param samples образцы
     * @param w       ширина масштабирования
     * @param h       высота масштабирования
     * @return образцы, для которых признаки успешно извлечены
     */
    public static List<Sample> extractAll(List<Sample> samples, int w, int h) {
        ImageLoader loader = new ImageLoader();
        FeatureExtractor fx = new FeatureExtractor();
        List<Sample> ok = new ArrayList<>();
        for (Sample s : samples) {
            try {
                BufferedImage img = loader.loadPrepared(s.getFile(), w, h);
                s.setFeatures(fx.extract(img));
                ok.add(s);
            } catch (ImageNotFoundException | NotAnImageException e) {
                System.err.println("Предупреждение: образец пропущен — " + e.getMessage());
            }
        }
        return ok;
    }

    /**
     * Выбирает рабочий словарь признаков для разбиения с классом сомнения (см. описание класса).
     *
     * @param train обучающая выборка с признаками
     * @return имена выбранных признаков
     * @throws IllegalStateException если нет образцов одного из классов
     */
    public List<String> selectFeatures(List<Sample> train) {
        return selectFeatures(train, false);
    }

    /**
     * Выбирает 3–5 признаков: пул кандидатов по критерию Фишера, затем жадный прямой отбор по
     * сбалансированной точности при кросс-валидации. Признаки добавляются, пока прирост не меньше
     * {@value #MIN_GAIN} (первые {@value #MIN_FEATURES} добавляются всегда).
     *
     * @param train обучающая выборка с признаками
     * @param tight {@code true} для разбиения без класса сомнения (A2)
     * @return имена выбранных признаков
     * @throws IllegalStateException если нет образцов одного из классов или нет информативных признаков
     */
    public List<String> selectFeatures(List<Sample> train, boolean tight) {
        Data d = Data.of(train);
        List<Integer> pool = candidatePool(d);
        if (pool.isEmpty()) {
            throw new IllegalStateException("Нет информативных признаков для обучения");
        }
        Folds folds = makeFolds(d);
        List<Integer> chosen = new ArrayList<>();
        double currentScore = Double.NEGATIVE_INFINITY;
        while (chosen.size() < MAX_FEATURES) {
            int bestCand = -1;
            double bestCandScore = Double.NEGATIVE_INFINITY;
            for (int cand : pool) {
                if (chosen.contains(cand)) {
                    continue;
                }
                int[] cols = new int[chosen.size() + 1];
                for (int i = 0; i < chosen.size(); i++) {
                    cols[i] = chosen.get(i);
                }
                cols[cols.length - 1] = cand;
                double sc = cvScore(d, folds, cols, tight);
                if (sc > bestCandScore + 1e-12) {
                    bestCandScore = sc;
                    bestCand = cand;
                }
            }
            if (bestCand < 0) {
                break;
            }
            if (chosen.size() >= MIN_FEATURES && bestCandScore < currentScore + MIN_GAIN) {
                break;
            }
            chosen.add(bestCand);
            currentScore = bestCandScore;
        }
        List<String> names = FeatureExtractor.featureNames();
        List<String> selected = new ArrayList<>();
        for (int idx : chosen) {
            selected.add(names.get(idx));
        }
        return selected;
    }

    /**
     * Подбирает решающие границы для выбранных признаков (см. описание класса).
     *
     * @param train     обучающая выборка
     * @param features  рабочий словарь признаков
     * @param partition разбиение (с двумя классами обучается без класса сомнения)
     * @return модель с подобранными границами и правилом голосования
     * @throws IllegalArgumentException если признак неизвестен
     * @throws IllegalStateException    если нет образцов одного из классов
     */
    public ModelConfig fitBoundaries(List<Sample> train, List<String> features, ClassPartition partition) {
        Data d = Data.of(train);
        if (!hasBoth(d.y)) {
            throw new IllegalStateException("Для обучения нужны образцы обоих классов (apple и other)");
        }
        List<String> all = FeatureExtractor.featureNames();
        int[] cols = new int[features.size()];
        for (int k = 0; k < cols.length; k++) {
            cols[k] = all.indexOf(features.get(k));
            if (cols[k] < 0) {
                throw new IllegalArgumentException("Неизвестный признак: " + features.get(k));
            }
        }
        Params p = fitParams(d.x, d.y, cols, partition.size() < 3, SWEEPS_FINAL);
        Map<String, double[]> th = new LinkedHashMap<>();
        for (int j = 0; j < cols.length; j++) {
            th.put(features.get(j), new double[]{p.sLo[j], p.sHi[j], p.lLo[j], p.lHi[j]});
        }
        return new ModelConfig(ModelConfig.CURRENT_VERSION, partition, features, th,
                DEFAULT_SIZE, DEFAULT_SIZE, 0.3, 0.5, p.minStrict, p.rejectVotes);
    }

    /**
     * Самообучение для класса ΩA1_3 «похоже на яблоко»: уверенность класса калибруется по
     * доле настоящих яблок среди объектов обучающей выборки, попавших в этот класс:
     * {@code uMin = 0.3}, {@code uMax = 0.3 + 0.2·p}.
     *
     * @param cfg   модель
     * @param train обучающая выборка
     * @return модель с откалиброванной уверенностью (или исходная, если класс пуст)
     */
    public ModelConfig clusterUnsure(ModelConfig cfg, List<Sample> train) {
        Classifier clf = newClassifier(cfg);
        int total = 0;
        int apples = 0;
        for (Sample s : train) {
            if (s.getFeatures() != null && clf.applyThresholds(s.getFeatures()) == 2) {
                total++;
                if (s.getLabel() == Sample.APPLE) {
                    apples++;
                }
            }
        }
        if (total == 0) {
            return cfg;
        }
        double p = (double) apples / total;
        return cfg.withUnsureConfidence(0.3, 0.3 + 0.2 * p);
    }

    /**
     * Сохраняет модель в JSON ({@code models/model_A*.json}).
     *
     * @param config модель
     * @param path   путь к файлу
     * @throws IOException при ошибке записи
     */
    public void saveModel(ModelConfig config, String path) throws IOException {
        config.save(Path.of(path));
    }

    /**
     * Обучает A1, строит A2 (перенос худшего класса в класс с минимальным ΔG), обучает A2
     * (отдельный словарь признаков и границы без класса сомнения), вычисляет R для обоих,
     * выбирает лучшее разбиение и сохраняет модели, матрицы ошибок, {@code gains_G.csv} и
     * {@code R_summary.csv}. Критерий считается на тестовой выборке.
     *
     * @param dataDir    папка данных
     * @param modelsDir  папка моделей
     * @param resultsDir папка результатов
     * @return обученные модели [A1, A2]
     * @throws IOException при ошибках чтения данных или записи результатов
     */
    public List<ModelConfig> trainAllPartitions(File dataDir, Path modelsDir, Path resultsDir)
            throws IOException {
        List<Sample> samples = loadDataset(dataDir);
        List<List<Sample>> split = splitTrainTest(samples, TRAIN_RATIO);
        writeMetadata(dataDir, samples);
        List<Sample> train = extractAll(split.get(0), DEFAULT_SIZE, DEFAULT_SIZE);
        List<Sample> test = extractAll(split.get(1), DEFAULT_SIZE, DEFAULT_SIZE);
        if (test.isEmpty()) {
            test = train;
        }
        EfficiencyCriterion crit = new EfficiencyCriterion();

        ModelConfig m1 = fitPartition(train, ClassPartition.A1);
        Evaluator ev1 = new Evaluator();
        ev1.evaluate(newClassifier(m1), test);
        EfficiencyCriterion.Analysis a1 = crit.analyze("A1", ev1.getLast().getMatrix());

        ClassPartition p2 = ClassPartition.A1.merge(a1.worstClass, a1.mergeTarget);
        ModelConfig m2 = fitPartition(train, p2);
        Evaluator ev2 = new Evaluator();
        ev2.evaluate(newClassifier(m2), test);
        crit.analyze("A2", ev2.getLast().getMatrix());

        saveModel(m1, modelsDir.resolve("model_A1.json").toString());
        saveModel(m2, modelsDir.resolve("model_A2.json").toString());
        ev1.saveConfusionMatrix(resultsDir.resolve("A1_confusion_matrix.csv").toString());
        ev2.saveConfusionMatrix(resultsDir.resolve("A2_confusion_matrix.csv").toString());
        crit.saveGains(resultsDir.resolve("gains_G.csv").toString());
        crit.saveSummary(resultsDir.resolve("R_summary.csv").toString());
        return List.of(m1, m2);
    }

    /* ------------------------------ внутренние алгоритмы ------------------------------ */

    /**
     * Пул кандидатов: признаки с наибольшим критерием Фишера (без исключённых и константных).
     *
     * @param d данные
     * @return индексы признаков в порядке убывания критерия
     * @throws IllegalStateException если нет образцов одного из классов
     */
    private static List<Integer> candidatePool(Data d) {
        List<String> names = FeatureExtractor.featureNames();
        int n = names.size();
        double[] sum1 = new double[n];
        double[] sum0 = new double[n];
        double[] sq1 = new double[n];
        double[] sq0 = new double[n];
        int c1 = 0;
        int c0 = 0;
        for (int i = 0; i < d.y.length; i++) {
            boolean apple = d.y[i] == Sample.APPLE;
            if (apple) {
                c1++;
            } else {
                c0++;
            }
            for (int k = 0; k < n; k++) {
                double v = d.x[i][k];
                if (apple) {
                    sum1[k] += v;
                    sq1[k] += v * v;
                } else {
                    sum0[k] += v;
                    sq0[k] += v * v;
                }
            }
        }
        if (c1 == 0 || c0 == 0) {
            throw new IllegalStateException("Для обучения нужны образцы обоих классов (apple и other)");
        }
        double[] score = new double[n];
        List<Integer> order = new ArrayList<>();
        for (int k = 0; k < n; k++) {
            double m1 = sum1[k] / c1;
            double m0 = sum0[k] / c0;
            double v1 = Math.max(0, sq1[k] / c1 - m1 * m1);
            double v0 = Math.max(0, sq0[k] / c0 - m0 * m0);
            score[k] = (m1 - m0) * (m1 - m0) / (v1 + v0 + 1e-9);
            boolean constant = v1 + v0 < 1e-12 && Math.abs(m1 - m0) < 1e-9;
            if (!EXCLUDED.contains(names.get(k)) && !constant) {
                order.add(k);
            }
        }
        order.sort((a, b) -> Double.compare(score[b], score[a]));
        return new ArrayList<>(order.subList(0, Math.min(CANDIDATE_POOL, order.size())));
    }

    /**
     * Строит стратифицированные фолды кросс-валидации (детерминированно).
     *
     * @param d данные
     * @return фолды
     */
    private static Folds makeFolds(Data d) {
        int n = d.y.length;
        int[] foldOf = new int[n];
        for (int label = 0; label < 2; label++) {
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (d.y[i] == label) {
                    idx.add(i);
                }
            }
            Collections.shuffle(idx, new Random(SEED + 100 + label));
            for (int t = 0; t < idx.size(); t++) {
                foldOf[idx.get(t)] = t % CV_FOLDS;
            }
        }
        Folds f = new Folds();
        for (int fold = 0; fold < CV_FOLDS; fold++) {
            int va = 0;
            for (int i = 0; i < n; i++) {
                if (foldOf[i] == fold) {
                    va++;
                }
            }
            int tr = n - va;
            f.trainX[fold] = new double[tr][];
            f.trainY[fold] = new int[tr];
            f.valX[fold] = new double[va][];
            f.valY[fold] = new int[va];
            int ti = 0;
            int vi = 0;
            for (int i = 0; i < n; i++) {
                if (foldOf[i] == fold) {
                    f.valX[fold][vi] = d.x[i];
                    f.valY[fold][vi++] = d.y[i];
                } else {
                    f.trainX[fold][ti] = d.x[i];
                    f.trainY[fold][ti++] = d.y[i];
                }
            }
        }
        return f;
    }

    /**
     * Средняя сбалансированная точность на валидационных фолдах для набора признаков.
     *
     * @param d     все обучающие данные
     * @param folds фолды
     * @param cols  индексы признаков
     * @param tight режим без класса сомнения
     * @return средняя сбалансированная точность
     */
    private static double cvScore(Data d, Folds folds, int[] cols, boolean tight) {
        double sum = 0;
        int used = 0;
        for (int fold = 0; fold < CV_FOLDS; fold++) {
            if (!hasBoth(folds.trainY[fold]) || folds.valY[fold].length == 0) {
                continue;
            }
            Params p = fitParams(folds.trainX[fold], folds.trainY[fold], cols, tight, SWEEPS_CV);
            sum += score(p, folds.valX[fold], folds.valY[fold]);
            used++;
        }
        if (used == 0) {
            Params p = fitParams(d.x, d.y, cols, tight, SWEEPS_CV);
            return score(p, d.x, d.y);
        }
        return sum / used;
    }

    /**
     * Проверяет, что среди меток есть оба класса.
     *
     * @param y метки
     * @return {@code true}, если есть и яблоки, и не-яблоки
     */
    private static boolean hasBoth(int[] y) {
        boolean apple = false;
        boolean other = false;
        for (int v : y) {
            if (v == Sample.APPLE) {
                apple = true;
            } else {
                other = true;
            }
        }
        return apple && other;
    }

    /**
     * Подбирает параметры модели для заданных признаков: начальные интервалы по 5–95 % квантилям
     * значений яблок, координатный спуск по границам, перебор параметров правила и запаса,
     * повторный спуск и повторный перебор.
     *
     * @param x      векторы признаков
     * @param y      метки
     * @param cols   индексы признаков
     * @param tight  режим без класса сомнения
     * @param sweeps число проходов координатного спуска
     * @return подобранные параметры
     */
    private static Params fitParams(double[][] x, int[] y, int[] cols, boolean tight, int sweeps) {
        int k = cols.length;
        double[] sd = new double[k];
        double[][] cuts = new double[k][];
        double[] sLo = new double[k];
        double[] sHi = new double[k];
        int apples = 0;
        for (int v : y) {
            if (v == Sample.APPLE) {
                apples++;
            }
        }
        for (int j = 0; j < k; j++) {
            double[] all = new double[x.length];
            double[] app = new double[apples];
            int ai = 0;
            double mean = 0;
            for (int i = 0; i < x.length; i++) {
                double v = x[i][cols[j]];
                all[i] = v;
                mean += v;
                if (y[i] == Sample.APPLE) {
                    app[ai++] = v;
                }
            }
            mean /= Math.max(1, x.length);
            double var = 0;
            for (double v : all) {
                var += (v - mean) * (v - mean);
            }
            sd[j] = Math.sqrt(var / Math.max(1, x.length));
            Arrays.sort(all);
            Arrays.sort(app);
            double[] c = new double[CUT_POINTS + 1];
            for (int t = 0; t <= CUT_POINTS; t++) {
                c[t] = quantile(all, t / (double) CUT_POINTS);
            }
            cuts[j] = Arrays.stream(c).distinct().toArray();
            sLo[j] = quantile(app, 0.05);
            sHi[j] = quantile(app, 0.95);
        }
        Params p = new Params(cols.clone(), sLo, sHi, sd, tight);
        descend(p, x, y, cuts, sweeps);
        tuneRule(p, x, y);
        descend(p, x, y, cuts, Math.max(1, sweeps / 2));
        tuneRule(p, x, y);
        return p;
    }

    /**
     * Координатный спуск по границам строгих интервалов: для каждой границы перебираются
     * кандидаты-квантили значений обоих классов, принимается значение с наибольшей
     * сбалансированной точностью (только при строгом улучшении).
     *
     * @param p      параметры (изменяются)
     * @param x      векторы признаков
     * @param y      метки
     * @param cuts   кандидаты границ по признакам
     * @param sweeps максимальное число проходов
     */
    private static void descend(Params p, double[][] x, int[] y, double[][] cuts, int sweeps) {
        double best = score(p, x, y);
        int k = p.cols.length;
        for (int sweep = 0; sweep < sweeps; sweep++) {
            boolean improved = false;
            for (int j = 0; j < k; j++) {
                double minWidth = 0.05 * p.sd[j];
                for (int side = 0; side < 2; side++) {
                    double bestVal = side == 0 ? p.sLo[j] : p.sHi[j];
                    for (double c : cuts[j]) {
                        if (side == 0) {
                            if (c > p.sHi[j] - minWidth) {
                                continue;
                            }
                            p.sLo[j] = c;
                        } else {
                            if (c < p.sLo[j] + minWidth) {
                                continue;
                            }
                            p.sHi[j] = c;
                        }
                        p.refresh();
                        double s = score(p, x, y);
                        if (s > best + 1e-12) {
                            best = s;
                            bestVal = c;
                            improved = true;
                        }
                    }
                    if (side == 0) {
                        p.sLo[j] = bestVal;
                    } else {
                        p.sHi[j] = bestVal;
                    }
                    p.refresh();
                }
            }
            if (!improved) {
                break;
            }
        }
    }

    /**
     * Перебирает параметры правила (a — число признаков в строгих интервалах для «яблока»,
     * b — число признаков вне широких интервалов для «явно не яблока») и запас широких интервалов.
     * Критерий: сначала сбалансированная точность, затем чистота класса ΩA1_2. Для режима без
     * класса сомнения a = k − b + 1, запас равен нулю.
     *
     * @param p параметры (изменяются)
     * @param x векторы признаков
     * @param y метки
     */
    private static void tuneRule(Params p, double[][] x, int[] y) {
        int k = p.cols.length;
        double bestScore = Double.NEGATIVE_INFINITY;
        double bestPurity = Double.NEGATIVE_INFINITY;
        int bestA = p.minStrict;
        int bestB = p.rejectVotes;
        double bestM = p.margin;
        for (int b = 1; b <= Math.min(2, k); b++) {
            int[] aOptions;
            if (p.tight) {
                aOptions = new int[]{Math.max(1, k - b + 1)};
            } else if (k > 1) {
                aOptions = new int[]{k, k - 1};
            } else {
                aOptions = new int[]{k};
            }
            double[] mOptions = p.tight ? new double[]{0.0} : LOOSE_MARGINS;
            for (int a : aOptions) {
                for (double m : mOptions) {
                    p.minStrict = a;
                    p.rejectVotes = b;
                    p.margin = m;
                    p.refresh();
                    double sc = score(p, x, y);
                    double pu = p.tight ? 0.0 : purity(p, x, y);
                    if (sc > bestScore + 1e-12 || (sc >= bestScore - 1e-12 && pu > bestPurity + 1e-12)) {
                        bestScore = sc;
                        bestPurity = pu;
                        bestA = a;
                        bestB = b;
                        bestM = m;
                    }
                }
            }
        }
        p.minStrict = bestA;
        p.rejectVotes = bestB;
        p.margin = bestM;
        p.refresh();
    }

    /**
     * Сбалансированная точность итогового решения «яблоко / не яблоко»:
     * {@code (TP/(TP+FN) + TN/(TN+FP)) / 2}. Решение «яблоко» принимается, если
     * r &lt; b и s ≥ a (см. {@link Classifier}).
     *
     * @param p параметры
     * @param x векторы признаков
     * @param y метки
     * @return значение 0..1
     */
    private static double score(Params p, double[][] x, int[] y) {
        int tp = 0;
        int fn = 0;
        int tn = 0;
        int fp = 0;
        int k = p.cols.length;
        for (int i = 0; i < x.length; i++) {
            double[] row = x[i];
            int s = 0;
            int r = 0;
            for (int j = 0; j < k; j++) {
                double v = row[p.cols[j]];
                if (v >= p.sLo[j] && v <= p.sHi[j]) {
                    s++;
                }
                if (v < p.lLo[j] || v > p.lHi[j]) {
                    r++;
                }
            }
            boolean apple = r < p.rejectVotes && s >= p.minStrict;
            if (y[i] == Sample.APPLE) {
                if (apple) {
                    tp++;
                } else {
                    fn++;
                }
            } else {
                if (apple) {
                    fp++;
                } else {
                    tn++;
                }
            }
        }
        double sum = 0;
        int c = 0;
        if (tp + fn > 0) {
            sum += (double) tp / (tp + fn);
            c++;
        }
        if (tn + fp > 0) {
            sum += (double) tn / (tn + fp);
            c++;
        }
        return c == 0 ? 0.0 : sum / c;
    }

    /**
     * Чистота класса ΩA1_2 «явно не яблоко»: доля «не яблок» в нём минус удвоенная доля яблок.
     *
     * @param p параметры
     * @param x векторы признаков
     * @param y метки
     * @return значение от −2 до 1
     */
    private static double purity(Params p, double[][] x, int[] y) {
        int apples = 0;
        int others = 0;
        int applesRejected = 0;
        int othersRejected = 0;
        int k = p.cols.length;
        for (int i = 0; i < x.length; i++) {
            int r = 0;
            for (int j = 0; j < k; j++) {
                double v = x[i][p.cols[j]];
                if (v < p.lLo[j] || v > p.lHi[j]) {
                    r++;
                }
            }
            boolean rejected = r >= p.rejectVotes;
            if (y[i] == Sample.APPLE) {
                apples++;
                if (rejected) {
                    applesRejected++;
                }
            } else {
                others++;
                if (rejected) {
                    othersRejected++;
                }
            }
        }
        double po = others == 0 ? 0.0 : (double) othersRejected / others;
        double pa = apples == 0 ? 0.0 : (double) applesRejected / apples;
        return po - 2.0 * pa;
    }

    /**
     * Квантиль отсортированной выборки с линейной интерполяцией.
     *
     * @param sorted отсортированные значения
     * @param p      уровень 0..1
     * @return значение квантили
     */
    private static double quantile(double[] sorted, double p) {
        double pos = p * (sorted.length - 1);
        int lo = (int) Math.floor(pos);
        int hi = (int) Math.ceil(pos);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
    }

    /**
     * Создаёт классификатор, преобразуя проверяемое исключение в непроверяемое
     * (внутри обучения модель всегда корректна).
     *
     * @param cfg модель
     * @return классификатор
     */
    private static Classifier newClassifier(ModelConfig cfg) {
        try {
            return new Classifier(cfg);
        } catch (ModelException e) {
            throw new IllegalStateException("Внутренняя ошибка модели: " + e.getMessage(), e);
        }
    }
}
