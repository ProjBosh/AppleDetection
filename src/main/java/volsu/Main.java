// src/main/java/volsu/Main.java
package volsu;

import volsu.Classifier.ClassificationResult;
import volsu.Evaluator.EvaluationResult;
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
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

/**
 * Точка входа CLI системы распознавания «яблоко / не яблоко».
 * <p>
 * Команды (в интерактивном режиме начинаются со слова {@code detect}; при запуске с аргументами
 * слово {@code detect} подставляется автоматически):
 * <ul>
 *   <li>{@code detect --image <image>} / {@code detect --model <model.json> <image>};</li>
 *   <li>{@code detect --train <dataDir> --out <model.json> [--partition A1|A2|ALL]};</li>
 *   <li>{@code detect --evaluate <dataDir> --model <model.json> [--partition A1|A2]};</li>
 *   <li>{@code detect --criterion <resultsDir>};</li>
 *   <li>{@code detect --help}.</li>
 * </ul>
 * Коды возврата: 0 — успех, 1 — ошибка ввода, 2 — ошибка модели.
 */
public final class Main {

    /** Код возврата: успех. */
    public static final int EXIT_OK = 0;
    /** Код возврата: ошибка ввода (файл не найден, не изображение, неверные параметры). */
    public static final int EXIT_INPUT_ERROR = 1;
    /** Код возврата: ошибка модели. */
    public static final int EXIT_MODEL_ERROR = 2;

    private static final String DEFAULT_MODEL = "models/model_A1.json";
    private static final String MODELS_DIR = "models";
    private static final String RESULTS_DIR = "results";
    private static final double LOW_CONFIDENCE = 0.55;
    private static final long TIME_LIMIT_NANOS = 1_000_000_000L;
    private static final Set<String> OPTIONS_WITH_VALUE = Set.of(
        "--image",
        "--model",
        "--train",
        "--out",
        "--evaluate",
        "--criterion",
        "--partition"
    );

    /**
     * Точка входа: перехватывает исключения и завершает работу через {@code System.exit(код)}.
     * Без аргументов запускается интерактивный режим (выход по команде {@code exit}).
     *
     * @param args аргументы командной строки
     */
    public static void main(String[] args) {
        int code = EXIT_OK;
        try {
            if (args.length > 0) {
                StringBuilder sb = new StringBuilder();
                for (String a : args) {
                    sb.append('"').append(a).append("\" ");
                }
                String line = sb.toString().trim();
                if (!args[0].equals("detect")) {
                    line = "detect " + line;
                }
                code = parseCommands(line);
            } else {
                try (Scanner scanner = new Scanner(System.in)) {
                    boolean getOut = false;
                    System.out.println("Enter the text:");
                    while (!getOut && scanner.hasNextLine()) {
                        String command = scanner.nextLine().trim();
                        if (command.equals("exit")) {
                            getOut = true;
                        } else if (!command.isEmpty()) {
                            parseCommands(command);
                        }
                        System.out.println("---\n");
                    }
                }
                System.out.println("The end!");
            }
        } catch (Exception e) {
            System.err.println("Непредвиденная ошибка: " + e);
            code = EXIT_INPUT_ERROR;
        }
        System.exit(code);
    }

    /**
     * Разбирает и выполняет одну команду.
     *
     * @param command строка команды, начинающаяся с {@code detect}
     * @return код возврата (0/1/2)
     */
    public static int parseCommands(String command) {
        List<String> tokens = tokenize(command);
        if (tokens.isEmpty()) {
            return EXIT_OK;
        }
        if (!tokens.get(0).equals("detect")) {
            System.err.println("Корректная команда должна начинаться с detect. Воспользуйтесь справкой через detect --help");
            return EXIT_INPUT_ERROR;
        }
        if (tokens.size() < 2) {
            System.err.println("Минимальное количество параметров равно 2. Воспользуйтесь справкой через detect --help");
            return EXIT_INPUT_ERROR;
        }
        Map<String, String> opts = new HashMap<>();
        String positional = null;
        boolean help = false;
        for (int i = 1; i < tokens.size(); i++) {
            String t = tokens.get(i);
            if (t.equals("--help")) {
                help = true;
            } else if (OPTIONS_WITH_VALUE.contains(t)) {
                if (i + 1 >= tokens.size() || tokens.get(i + 1).startsWith("--")) {
                    System.err.println("Для опции " + t + " не указано значение");
                    return EXIT_INPUT_ERROR;
                }
                opts.put(t, tokens.get(++i));
            } else if (t.startsWith("--")) {
                System.err.println("Неизвестная опция: " + t + ". Воспользуйтесь справкой через detect --help");
                return EXIT_INPUT_ERROR;
            } else if (positional == null) {
                positional = t;
            } else {
                System.err.println("Лишний параметр: " + t);
                return EXIT_INPUT_ERROR;
            }
        }

        // Обработка опций
        if (help) {
            printUsage();
            return EXIT_OK;
        }
        if (opts.containsKey("--train")) {
            String part = opts.getOrDefault("--partition", "A1");
            String out = opts.get("--out");
            if (out == null) {
                out = MODELS_DIR + "/model_" + part.toUpperCase(Locale.ROOT) + ".json";
            }
            return runTrain(opts.get("--train"), out, part);
        }
        if (opts.containsKey("--evaluate")) {
            return runEvaluate(opts.get("--evaluate"), opts.getOrDefault("--model", DEFAULT_MODEL),
                    opts.get("--partition"));
        }
        if (opts.containsKey("--criterion")) {
            return runCriterion(opts.get("--criterion"));
        }
        String image = opts.get("--image") != null ? opts.get("--image") : positional;
        if (image == null) {
            System.err.println("Не указано изображение. Воспользуйтесь справкой через detect --help");
            return EXIT_INPUT_ERROR;
        }
        return runClassify(image, opts.get("--model"));
    }

    /**
     * Сценарий распознавания одного изображения.
     *
     * @param imagePath путь к изображению
     * @param modelPath путь к модели или {@code null} (тогда {@code models/model_A1.json})
     * @return код возврата
     */
    private static int runClassify(String imagePath, String modelPath) {
        long t0 = System.nanoTime();
        try {
            ModelConfig config = ModelConfig.load(Path.of(expandPath(modelPath == null ? DEFAULT_MODEL : modelPath)));
            Classifier clf = new Classifier(config);
            ImageLoader loader = new ImageLoader();
            BufferedImage img = loader.loadPrepared(new File(expandPath(imagePath)),
                    config.getResizeWidth(), config.getResizeHeight());
            double[] features = new FeatureExtractor().extract(img);
            printResult(clf.classify(features));
            if (System.nanoTime() - t0 > TIME_LIMIT_NANOS) {
                System.err.println("Предупреждение: время обработки превысило 1 с (ограничение C0)");
            }
            return EXIT_OK;
        } catch (ModelException e) {
            System.err.println("Ошибка модели: " + e.getMessage());
            return EXIT_MODEL_ERROR;
        } catch (ImageNotFoundException | NotAnImageException e) {
            System.err.println("Ошибка ввода: " + e.getMessage());
            return EXIT_INPUT_ERROR;
        }
    }

    /**
     * Сценарий обучения. Значение {@code ALL} в качестве разбиения запускает полный цикл
     * (A1 → A2 → выбор лучшего по R) с записью моделей и результатов в {@code models/} и {@code results/}.
     *
     * @param dataDir      папка данных
     * @param modelOutPath путь для сохранения модели
     * @param partitionName имя разбиения (A1, A2 или ALL)
     * @return код возврата
     */
    private static int runTrain(String dataDir, String modelOutPath, String partitionName) {
        Trainer trainer = new Trainer();
        File dir = new File(expandPath(dataDir));
        if (partitionName.equalsIgnoreCase("ALL")) {
            try {
                List<ModelConfig> models = trainer.trainAllPartitions(dir, Path.of(MODELS_DIR), Path.of(RESULTS_DIR));
                System.out.println("Обучены модели: " + models.size() + "; результаты в папке " + RESULTS_DIR);
                return EXIT_OK;
            } catch (IOException | IllegalStateException e) {
                System.err.println("Ошибка обучения: " + e.getMessage());
                return EXIT_INPUT_ERROR;
            }
        }
        ModelConfig config;
        try {
            config = trainer.train(dir, ClassPartition.fromName(partitionName));
        } catch (IOException | IllegalArgumentException | IllegalStateException e) {
            System.err.println("Ошибка данных: " + e.getMessage());
            return EXIT_INPUT_ERROR;
        }
        try {
            trainer.saveModel(config, expandPath(modelOutPath));
        } catch (IOException e) {
            System.err.println("Ошибка сохранения модели: " + e.getMessage());
            return EXIT_MODEL_ERROR;
        }
        System.out.println("Модель сохранена: " + modelOutPath + "; признаки: " + config.getFeatures());
        return EXIT_OK;
    }

    /**
     * Сценарий оценки модели на тестовой выборке; матрица ошибок сохраняется в
     * {@code results/<разбиение>_confusion_matrix.csv}.
     *
     * @param dataDir       папка данных
     * @param modelPath     путь к модели
     * @param partitionName ожидаемое разбиение или {@code null}
     * @return код возврата
     */
    private static int runEvaluate(String dataDir, String modelPath, String partitionName) {
        try {
            ModelConfig config = ModelConfig.load(Path.of(expandPath(modelPath)));
            Classifier clf = new Classifier(config);
            String actual = config.getPartition().name();
            if (partitionName != null && !partitionName.equalsIgnoreCase(actual)) {
                System.err.println("Предупреждение: модель построена для разбиения " + actual
                        + ", опция --partition " + partitionName + " проигнорирована");
            }
            List<Sample> samples = Trainer.loadDataset(new File(expandPath(dataDir)));
            List<Sample> test = Trainer.splitTrainTest(samples, Trainer.TRAIN_RATIO).get(1);
            if (test.isEmpty()) {
                test = samples;
            }
            Evaluator evaluator = new Evaluator();
            EvaluationResult result = evaluator.evaluate(clf, test);
            evaluator.printReport();
            evaluator.saveConfusionMatrix(RESULTS_DIR + "/" + actual + "_confusion_matrix.csv");
            System.out.println(String.format(Locale.ROOT, "Оценено образцов: %d", result.getMatrix().total()));
            return EXIT_OK;
        } catch (ModelException e) {
            System.err.println("Ошибка модели: " + e.getMessage());
            return EXIT_MODEL_ERROR;
        } catch (IOException | IllegalArgumentException e) {
            System.err.println("Ошибка ввода: " + e.getMessage());
            return EXIT_INPUT_ERROR;
        }
    }

    /**
     * Сценарий расчёта критерия эффективности: читает {@code A<k>_confusion_matrix.csv},
     * вычисляет G, ΔG, R, выбирает лучшее разбиение и пишет {@code gains_G.csv}, {@code R_summary.csv}.
     *
     * @param resultsDir папка результатов
     * @return код возврата
     */
    private static int runCriterion(String resultsDir) {
        Path dir = Path.of(expandPath(resultsDir));
        EfficiencyCriterion crit = new EfficiencyCriterion();
        try {
            int found = 0;
            for (int k = 1; k <= 9; k++) {
                String name = "A" + k;
                Path p = dir.resolve(name + "_confusion_matrix.csv");
                if (!Files.exists(p)) {
                    continue;
                }
                ConfusionMatrix cm = ConfusionMatrix.fromCsv(name, Files.readAllLines(p, StandardCharsets.UTF_8));
                EfficiencyCriterion.Analysis a = crit.analyze(name, cm);
                System.out.println(String.format(Locale.ROOT, "R(%s) = %.4f", name, a.r));
                found++;
            }
            if (found == 0) {
                System.err.println("В папке " + dir + " не найдены файлы A*_confusion_matrix.csv");
                return EXIT_INPUT_ERROR;
            }
            crit.saveGains(dir.resolve("gains_G.csv").toString());
            crit.saveSummary(dir.resolve("R_summary.csv").toString());
            EfficiencyCriterion.Analysis best = crit.getResults().get(crit.bestIndex());
            System.out.println(String.format(Locale.ROOT, "Лучшее разбиение: %s (R = %.4f)", best.name, best.r));
            return EXIT_OK;
        } catch (IOException | IllegalArgumentException e) {
            System.err.println("Ошибка ввода: " + e.getMessage());
            return EXIT_INPUT_ERROR;
        }
    }

    /**
     * Выводит результат: «Яблоко (уверенность 0.92)», «Не яблоко (уверенность 0.87)» либо
     * «Не яблоко» при низкой уверенности (&lt; {@value #LOW_CONFIDENCE}).
     *
     * @param result результат классификации
     */
    private static void printResult(ClassificationResult result) {
        if (result.isApple()) {
            System.out.println(String.format(Locale.ROOT, "Яблоко (уверенность %.2f)", result.getConfidence()));
        } else if (result.getConfidence() >= LOW_CONFIDENCE) {
            System.out.println(String.format(Locale.ROOT, "Не яблоко (уверенность %.2f)", result.getConfidence()));
        } else {
            System.out.println("Не яблоко");
        }
    }

    /** Выводит справку по использованию со всеми опциями. */
    private static void printUsage() {
        System.out.println(
                "\n-------------------------------------------------------------------\n"
                        + "Base options:\n"
                        + " * detect --image <image>\n"
                        + " * detect --model <model.json> <image>\n"
                        + " * detect --train <dataDir> --out <model.json> [--partition A1|A2|ALL]\n"
                        + " * detect --evaluate <dataDir> --model <model.json> [--partition A1|A2]\n"
                        + " * detect --criterion <resultsDir>\n"
                        + " * detect --help\n"
                        + "Codes: 0 - ok, 1 - input error, 2 - model error\n"
                        + "-------------------------------------------------------------------\n");
    }

    /**
     * Раскрывает {@code ~} в начале пути до домашнего каталога пользователя.
     *
     * @param path исходный путь
     * @return путь с раскрытым {@code ~}
     */
    private static String expandPath(String path) {
        if (path.equals("~") || path.startsWith("~/")) {
            return System.getProperty("user.home") + path.substring(1);
        }
        return path;
    }

    /**
     * Разбивает строку на токены по пробелам с учётом двойных кавычек.
     *
     * @param s строка команды
     * @return список токенов
     */
    private static List<String> tokenize(String s) {
        List<String> out = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQuotes = false;
        boolean has = false;
        for (char c : s.toCharArray()) {
            if (c == '"') {
                inQuotes = !inQuotes;
                has = true;
            } else if (Character.isWhitespace(c) && !inQuotes) {
                if (has || cur.length() > 0) {
                    out.add(cur.toString());
                    cur.setLength(0);
                    has = false;
                }
            } else {
                cur.append(c);
            }
        }
        if (has || cur.length() > 0) {
            out.add(cur.toString());
        }
        return out;
    }
}
