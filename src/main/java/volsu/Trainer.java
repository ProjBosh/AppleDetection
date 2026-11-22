// src/main/java/volsu/Trainer.java
package volsu;

import volsu.model.ClassPartition;
import volsu.model.ConfusionMatrix;
import volsu.model.ModelConfig;
import volsu.model.Sample;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
 * Информативность признака оценивается критерием Фишера
 * {@code F = (μ₁ − μ₀)² / (σ₁² + σ₀²)} (класс 1 — яблоки, класс 0 — остальное).
 * Границы подбираются перебором параметров по сетке (доля отсечения q строгого интервала и
 * относительное расширение m широкого интервала) с максимизацией критерия R на обучающей выборке.
 */
public class Trainer {

    /** Доля обучающей выборки. */
    public static final double TRAIN_RATIO = 0.7;
    /** Зерно генератора для воспроизводимого разбиения. */
    public static final long SEED = 42L;
    /** Размер рабочего словаря признаков. */
    public static final int SELECTED_FEATURES = 4;
    /** Размер масштабирования по умолчанию. */
    public static final int DEFAULT_SIZE = 256;

    private static final Set<String> IMAGE_EXT = Set.of("jpg", "jpeg", "png", "bmp", "gif");
    /** Признаки, не пригодные для интервальных границ (циклический тон; дублирующее соотношение сторон). */
    private static final Set<String> EXCLUDED = Set.of("meanHue", "aspectRatio");
    private static final double[] LOOSE_MARGINS = {0.25, 0.1, 0.5, 0.0};
    private static final double[] STRICT_QUANTILES = {0.05, 0.0, 0.02, 0.1, 0.15};

    private final EfficiencyCriterion criterion = new EfficiencyCriterion();

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
        writeMetadataIfAbsent(dataDir, samples);
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
        List<String> features = selectFeatures(train);
        ModelConfig cfg = fitBoundaries(train, features, partition);
        return clusterUnsure(cfg, train);
    }

    /**
     * Читает выборку. Если рядом с папкой данных есть {@code python/metadata.csv}
     * (путь, класс, split), используется он; иначе сканируются {@code data/apple} и {@code data/other}.
     *
     * @param dataDir папка данных
     * @return список образцов
     * @throws IOException если папка/подпапки отсутствуют или выборка пуста
     */
    public static List<Sample> loadDataset(File dataDir) throws IOException {
        if (dataDir == null || !dataDir.isDirectory()) {
            throw new IOException("Папка с данными не найдена: " + dataDir);
        }
        File root = dataDir.getAbsoluteFile().getParentFile();
        File meta = root == null ? null : new File(root, "python/metadata.csv");
        List<Sample> samples = new ArrayList<>();
        if (meta != null && meta.isFile()) {
            List<String> lines = Files.readAllLines(meta.toPath(), StandardCharsets.UTF_8);
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
                String split = p[2].trim().toLowerCase(Locale.ROOT);
                samples.add(new Sample(f, label, split.isEmpty() ? null : split));
            }
        }
        if (samples.isEmpty()) {
            scan(new File(dataDir, "apple"), Sample.APPLE, samples);
            scan(new File(dataDir, "other"), Sample.OTHER, samples);
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
     * Если у всех образцов уже назначена подвыборка (из metadata.csv), используется она.
     *
     * @param samples образцы
     * @param ratio   доля обучающей выборки
     * @return список из двух списков: [0] — train, [1] — test
     */
    public static List<List<Sample>> splitTrainTest(List<Sample> samples, double ratio) {
        List<Sample> train = new ArrayList<>();
        List<Sample> test = new ArrayList<>();
        boolean assigned = true;
        for (Sample s : samples) {
            if (s.getSplit() == null) {
                assigned = false;
                break;
            }
        }
        if (assigned) {
            for (Sample s : samples) {
                (s.getSplit().equals("test") ? test : train).add(s);
            }
        } else {
            for (int label = 0; label < 2; label++) {
                List<Sample> group = new ArrayList<>();
                for (Sample s : samples) {
                    if (s.getLabel() == label) {
                        group.add(s);
                    }
                }
                Collections.shuffle(group, new Random(SEED + label));
                int n = group.size();
                int cut = n <= 1 ? n : (int) Math.round(n * ratio);
                if (n > 1 && cut >= n) {
                    cut = n - 1;
                }
                for (int i = 0; i < n; i++) {
                    Sample s = group.get(i);
                    s.setSplit(i < cut ? "train" : "test");
                    (i < cut ? train : test).add(s);
                }
            }
        }
        List<List<Sample>> res = new ArrayList<>();
        res.add(train);
        res.add(test);
        return res;
    }

    /**
     * Записывает {@code python/metadata.csv} (путь, класс, split), если файла ещё нет.
     *
     * @param dataDir папка данных
     * @param samples образцы с назначенной подвыборкой
     * @throws IOException при ошибке записи
     */
    private static void writeMetadataIfAbsent(File dataDir, List<Sample> samples) throws IOException {
        File root = dataDir.getAbsoluteFile().getParentFile();
        if (root == null) {
            return;
        }
        File meta = new File(root, "python/metadata.csv");
        if (meta.exists()) {
            return;
        }
        StringBuilder sb = new StringBuilder("path,class,split\n");
        for (Sample s : samples) {
            String rel = root.toPath().relativize(s.getFile().getAbsoluteFile().toPath())
                    .toString().replace('\\', '/');
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
     * Выбирает 3–5 (по умолчанию {@value #SELECTED_FEATURES}) наиболее разделяющих признаков
     * по критерию Фишера. Циклический признак {@code meanHue} и {@code aspectRatio}
     * (его симметричная версия — {@code elongation}) исключаются.
     *
     * @param train обучающая выборка с признаками
     * @return имена выбранных признаков
     * @throws IllegalStateException если нет образцов одного из классов
     */
    public List<String> selectFeatures(List<Sample> train) {
        List<String> names = FeatureExtractor.featureNames();
        int n = names.size();
        double[] sum1 = new double[n];
        double[] sum0 = new double[n];
        double[] sq1 = new double[n];
        double[] sq0 = new double[n];
        int c1 = 0;
        int c0 = 0;
        for (Sample s : train) {
            double[] f = s.getFeatures();
            if (f == null) {
                continue;
            }
            boolean apple = s.getLabel() == Sample.APPLE;
            if (apple) {
                c1++;
            } else {
                c0++;
            }
            for (int k = 0; k < n; k++) {
                if (apple) {
                    sum1[k] += f[k];
                    sq1[k] += f[k] * f[k];
                } else {
                    sum0[k] += f[k];
                    sq0[k] += f[k] * f[k];
                }
            }
        }
        if (c1 == 0 || c0 == 0) {
            throw new IllegalStateException("Для обучения нужны образцы обоих классов (apple и other)");
        }
        double[] score = new double[n];
        for (int k = 0; k < n; k++) {
            double m1 = sum1[k] / c1;
            double m0 = sum0[k] / c0;
            double v1 = Math.max(0, sq1[k] / c1 - m1 * m1);
            double v0 = Math.max(0, sq0[k] / c0 - m0 * m0);
            score[k] = (m1 - m0) * (m1 - m0) / (v1 + v0 + 1e-9);
        }
        List<Integer> order = new ArrayList<>();
        for (int k = 0; k < n; k++) {
            if (!EXCLUDED.contains(names.get(k))) {
                order.add(k);
            }
        }
        order.sort((a, b) -> Double.compare(score[b], score[a]));
        List<String> selected = new ArrayList<>();
        for (int i = 0; i < Math.min(SELECTED_FEATURES, order.size()); i++) {
            selected.add(names.get(order.get(i)));
        }
        return selected;
    }

    /**
     * Подбирает решающие границы. Для выбранных признаков по значениям яблок строятся:
     * строгий интервал {@code [Q(q), Q(1 − q)]} и широкий интервал
     * {@code [min − m·R, max + m·R]}, где R = max − min. Перебираются q и m, выбирается пара
     * с максимальным R на обучающей выборке (при равенстве — первая в порядке перебора).
     *
     * @param train     обучающая выборка
     * @param features  рабочий словарь признаков
     * @param partition разбиение
     * @return модель с подобранными границами
     */
    public ModelConfig fitBoundaries(List<Sample> train, List<String> features, ClassPartition partition) {
        List<String> all = FeatureExtractor.featureNames();
        double[][] sorted = new double[features.size()][];
        for (int k = 0; k < features.size(); k++) {
            int idx = all.indexOf(features.get(k));
            List<Double> vals = new ArrayList<>();
            for (Sample s : train) {
                if (s.getLabel() == Sample.APPLE && s.getFeatures() != null) {
                    vals.add(s.getFeatures()[idx]);
                }
            }
            sorted[k] = vals.stream().mapToDouble(Double::doubleValue).sorted().toArray();
            if (sorted[k].length == 0) {
                throw new IllegalStateException("В обучающей выборке нет яблок");
            }
        }
        ModelConfig best = null;
        double bestR = Double.NEGATIVE_INFINITY;
        for (double m : LOOSE_MARGINS) {
            for (double q : STRICT_QUANTILES) {
                ModelConfig cand = buildConfig(partition, features, sorted, q, m);
                double r = criterion.computeR(Evaluator.buildMatrix(newClassifier(cand), train));
                if (r > bestR + 1e-12) {
                    bestR = r;
                    best = cand;
                }
            }
        }
        return best;
    }

    /**
     * Строит модель по квантилям значений яблок.
     *
     * @param partition разбиение
     * @param features  признаки
     * @param sorted    отсортированные значения яблок по признакам
     * @param q         доля отсечения с каждого края строгого интервала
     * @param margin    относительное расширение широкого интервала
     * @return модель
     */
    private ModelConfig buildConfig(ClassPartition partition, List<String> features,
                                    double[][] sorted, double q, double margin) {
        Map<String, double[]> th = new LinkedHashMap<>();
        for (int k = 0; k < features.size(); k++) {
            double[] v = sorted[k];
            double min = v[0];
            double max = v[v.length - 1];
            double range = Math.max(max - min, 1e-3);
            double sLo = quantile(v, q);
            double sHi = quantile(v, 1.0 - q);
            double lLo = Math.min(min - margin * range, sLo);
            double lHi = Math.max(max + margin * range, sHi);
            th.put(features.get(k), new double[]{sLo, sHi, lLo, lHi});
        }
        return new ModelConfig(partition, features, th, DEFAULT_SIZE, DEFAULT_SIZE);
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
     * Обучает A1, строит A2 (перенос худшего класса в класс с минимальным ΔG), обучает A2,
     * вычисляет R для обоих, выбирает лучшее разбиение и сохраняет модели, матрицы ошибок,
     * {@code gains_G.csv} и {@code R_summary.csv}. Критерий считается на тестовой выборке.
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
        writeMetadataIfAbsent(dataDir, samples);
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
