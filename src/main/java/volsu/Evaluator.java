// src/main/java/volsu/Evaluator.java
package volsu;

import volsu.model.ClassPartition;
import volsu.model.ConfusionMatrix;
import volsu.model.Sample;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Метрики качества распознавания: матрица ошибок по классам разбиения, бинарная матрица
 * итогового решения «яблоко / не яблоко», accuracy, precision, recall и вероятность
 * правильного решения задачи распознавания.
 * <p>
 * Формулы (яблоко — «положительный» исход): accuracy = (TP + TN)/(TP + TN + FP + FN),
 * precision = TP/(TP + FP), recall = TP/(TP + FN); при нулевом знаменателе значение 0.
 */
public class Evaluator {

    /**
     * Результат оценки.
     */
    public static final class EvaluationResult {
        private final ConfusionMatrix matrix;
        private final int tp;
        private final int tn;
        private final int fp;
        private final int fn;
        private final double accuracy;
        private final double precision;
        private final double recall;
        private final double probabilityCorrect;

        /**
         * Создаёт результат из матрицы ошибок.
         *
         * @param matrix матрица ошибок по классам
         */
        public EvaluationResult(ConfusionMatrix matrix) {
            this.matrix = matrix;
            int[] b = matrix.binary();
            this.tp = b[0];
            this.tn = b[1];
            this.fp = b[2];
            this.fn = b[3];
            this.accuracy = Evaluator.accuracy(tp, tn, fp, fn);
            this.precision = Evaluator.precision(tp, fp);
            this.recall = Evaluator.recall(tp, fn);
            this.probabilityCorrect = Evaluator.probabilityCorrect(tp, tn, fp, fn);
        }

        /**
         * Матрица ошибок по классам разбиения.
         *
         * @return матрица
         */
        public ConfusionMatrix getMatrix() {
            return matrix;
        }

        /**
         * Истинно положительные.
         *
         * @return TP
         */
        public int getTp() {
            return tp;
        }

        /**
         * Истинно отрицательные.
         *
         * @return TN
         */
        public int getTn() {
            return tn;
        }

        /**
         * Ложные срабатывания (не-яблоко принято за яблоко).
         *
         * @return FP
         */
        public int getFp() {
            return fp;
        }

        /**
         * Пропуски (яблоко принято за не-яблоко).
         *
         * @return FN
         */
        public int getFn() {
            return fn;
        }

        /**
         * Доля верных решений.
         *
         * @return accuracy
         */
        public double getAccuracy() {
            return accuracy;
        }

        /**
         * Точность.
         *
         * @return precision
         */
        public double getPrecision() {
            return precision;
        }

        /**
         * Полнота.
         *
         * @return recall
         */
        public double getRecall() {
            return recall;
        }

        /**
         * Вероятность правильного итогового решения.
         *
         * @return значение 0..1
         */
        public double getProbabilityCorrect() {
            return probabilityCorrect;
        }
    }

    private EvaluationResult last;

    /**
     * Прогоняет классификатор по тестовой выборке. Образцы без вектора признаков обрабатываются
     * через {@link Trainer#extractAll(List, int, int)}; нечитаемые изображения пропускаются.
     *
     * @param clf     классификатор
     * @param testSet тестовая выборка
     * @return результат оценки (также запоминается для {@link #printReport()})
     */
    public EvaluationResult evaluate(Classifier clf, List<Sample> testSet) {
        List<Sample> ready = new ArrayList<>();
        List<Sample> need = new ArrayList<>();
        for (Sample s : testSet) {
            if (s.getFeatures() == null) {
                need.add(s);
            } else {
                ready.add(s);
            }
        }
        if (!need.isEmpty()) {
            ready.addAll(Trainer.extractAll(need, clf.getConfig().getResizeWidth(),
                    clf.getConfig().getResizeHeight()));
        }
        last = new EvaluationResult(buildMatrix(clf, ready));
        return last;
    }

    /**
     * Строит матрицу ошибок для образцов с готовыми признаками.
     *
     * @param clf     классификатор
     * @param samples образцы (с заполненными признаками; остальные пропускаются)
     * @return матрица ошибок по классам разбиения классификатора
     */
    public static ConfusionMatrix buildMatrix(Classifier clf, List<Sample> samples) {
        ConfusionMatrix cm = new ConfusionMatrix(clf.getConfig().getPartition());
        for (Sample s : samples) {
            if (s.getFeatures() == null) {
                continue;
            }
            cm.add(s.getLabel(), clf.classify(s.getFeatures()).getClassIndex());
        }
        return cm;
    }

    /**
     * Строит матрицу ошибок по классам из готовых предсказаний.
     *
     * @param partition  разбиение
     * @param trueLabels истинные метки (0 — яблоко, 1 — не яблоко)
     * @param predicted  предсказанные классы разбиения
     * @return матрица ошибок (строки — истинные метки, столбцы — предсказанные классы)
     */
    public static ConfusionMatrix confusionMatrix(ClassPartition partition, int[] trueLabels, int[] predicted) {
        ConfusionMatrix cm = new ConfusionMatrix(partition);
        for (int i = 0; i < trueLabels.length; i++) {
            cm.add(trueLabels[i], predicted[i]);
        }
        return cm;
    }

    /**
     * Строит бинарную матрицу итогового решения «яблоко / не яблоко».
     *
     * @param partition  разбиение
     * @param trueLabels истинные метки
     * @param predicted  предсказанные классы
     * @return массив {TP, TN, FP, FN}
     */
    public static int[] binaryConfusionMatrix(ClassPartition partition, int[] trueLabels, int[] predicted) {
        return confusionMatrix(partition, trueLabels, predicted).binary();
    }

    /**
     * Accuracy = (TP + TN) / (TP + TN + FP + FN).
     *
     * @param tp истинно положительные
     * @param tn истинно отрицательные
     * @param fp ложно положительные
     * @param fn ложно отрицательные
     * @return доля верных решений; 0 для пустой выборки
     */
    public static double accuracy(int tp, int tn, int fp, int fn) {
        int total = tp + tn + fp + fn;
        return total == 0 ? 0.0 : (double) (tp + tn) / total;
    }

    /**
     * Precision = TP / (TP + FP).
     *
     * @param tp истинно положительные
     * @param fp ложно положительные
     * @return точность; 0 при нулевом знаменателе
     */
    public static double precision(int tp, int fp) {
        return tp + fp == 0 ? 0.0 : (double) tp / (tp + fp);
    }

    /**
     * Recall = TP / (TP + FN).
     *
     * @param tp истинно положительные
     * @param fn ложно отрицательные
     * @return полнота; 0 при нулевом знаменателе
     */
    public static double recall(int tp, int fn) {
        return tp + fn == 0 ? 0.0 : (double) tp / (tp + fn);
    }

    /**
     * Вероятность правильного решения задачи распознавания — доля верных итоговых решений
     * «яблоко / не яблоко» (оценка P по тестовой выборке).
     *
     * @param tp истинно положительные
     * @param tn истинно отрицательные
     * @param fp ложно положительные
     * @param fn ложно отрицательные
     * @return вероятность 0..1
     */
    public static double probabilityCorrect(int tp, int tn, int fp, int fn) {
        return accuracy(tp, tn, fp, fn);
    }

    /**
     * Результат последней оценки.
     *
     * @return результат или {@code null}, если оценка ещё не выполнялась
     */
    public EvaluationResult getLast() {
        return last;
    }

    /**
     * Записывает матрицу ошибок и метрики последней оценки в CSV
     * ({@code results/A*_confusion_matrix.csv}). Первые четыре строки — матрица
     * ({@link ConfusionMatrix#toCsv()}), далее строки {@code metric,<имя>,<значение>}.
     *
     * @param path путь к CSV
     * @throws IOException при ошибке записи
     * @throws IllegalStateException если оценка не выполнялась
     */
    public void saveConfusionMatrix(String path) throws IOException {
        EvaluationResult r = requireLast();
        Path p = Path.of(path);
        Path parent = p.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        StringBuilder sb = new StringBuilder(r.getMatrix().toCsv());
        sb.append(String.format(Locale.ROOT, "metric,accuracy,%.6f%n", r.getAccuracy()));
        sb.append(String.format(Locale.ROOT, "metric,precision,%.6f%n", r.getPrecision()));
        sb.append(String.format(Locale.ROOT, "metric,recall,%.6f%n", r.getRecall()));
        sb.append(String.format(Locale.ROOT, "metric,probabilityCorrect,%.6f%n", r.getProbabilityCorrect()));
        Files.writeString(p, sb.toString(), StandardCharsets.UTF_8);
    }

    /**
     * Выводит в консоль матрицу ошибок и метрики последней оценки.
     *
     * @throws IllegalStateException если оценка не выполнялась
     */
    public void printReport() {
        EvaluationResult r = requireLast();
        System.out.println("Матрица ошибок (" + r.getMatrix().partition().name() + "):");
        System.out.print(r.getMatrix().toCsv());
        System.out.println(String.format(Locale.ROOT, "TP=%d TN=%d FP=%d FN=%d", r.getTp(), r.getTn(), r.getFp(), r.getFn()));
        System.out.println(String.format(Locale.ROOT, "accuracy  = %.4f", r.getAccuracy()));
        System.out.println(String.format(Locale.ROOT, "precision = %.4f", r.getPrecision()));
        System.out.println(String.format(Locale.ROOT, "recall    = %.4f", r.getRecall()));
        System.out.println(String.format(Locale.ROOT, "P(верное решение) = %.4f", r.getProbabilityCorrect()));
    }

    /**
     * Возвращает последний результат или бросает исключение.
     *
     * @return результат
     */
    private EvaluationResult requireLast() {
        if (last == null) {
            throw new IllegalStateException("Оценка ещё не выполнялась");
        }
        return last;
    }
}
