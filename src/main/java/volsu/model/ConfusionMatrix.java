// src/main/java/volsu/model/ConfusionMatrix.java
package volsu.model;

import java.util.ArrayList;
import java.util.List;

/**
 * Матрица ошибок: строки — истинные метки (0 — яблоко, 1 — не яблоко),
 * столбцы — предсказанные классы разбиения {@link ClassPartition}.
 * Из неё получаются как матрица по классам, так и бинарная матрица итоговых решений.
 */
public final class ConfusionMatrix {

    private final ClassPartition partition;
    private final int[][] counts;

    /**
     * Создаёт пустую матрицу для разбиения.
     *
     * @param partition разбиение, определяющее столбцы
     */
    public ConfusionMatrix(ClassPartition partition) {
        this.partition = partition;
        this.counts = new int[2][partition.size()];
    }

    /**
     * Возвращает разбиение матрицы.
     *
     * @return разбиение
     */
    public ClassPartition partition() {
        return partition;
    }

    /**
     * Число столбцов (классов разбиения).
     *
     * @return m
     */
    public int classCount() {
        return partition.size();
    }

    /**
     * Добавляет один результат классификации.
     *
     * @param trueLabel      истинная метка (0/1)
     * @param predictedClass предсказанный класс разбиения
     */
    public void add(int trueLabel, int predictedClass) {
        add(trueLabel, predictedClass, 1);
    }

    /**
     * Добавляет несколько одинаковых результатов.
     *
     * @param trueLabel      истинная метка (0/1)
     * @param predictedClass предсказанный класс
     * @param count          количество
     */
    public void add(int trueLabel, int predictedClass, int count) {
        counts[trueLabel][predictedClass] += count;
    }

    /**
     * Значение ячейки.
     *
     * @param trueLabel      истинная метка
     * @param predictedClass предсказанный класс
     * @return число объектов
     */
    public int get(int trueLabel, int predictedClass) {
        return counts[trueLabel][predictedClass];
    }

    /**
     * Общее число объектов.
     *
     * @return сумма всех ячеек
     */
    public int total() {
        int t = 0;
        for (int[] row : counts) {
            for (int c : row) {
                t += c;
            }
        }
        return t;
    }

    /**
     * Бинарная матрица итоговых решений «яблоко / не яблоко».
     *
     * @return массив {TP, TN, FP, FN}, где «положительный» исход — решение l1 «яблоко»
     */
    public int[] binary() {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (int p = 0; p < classCount(); p++) {
            if (partition.decisionFor(p) == ClassPartition.DECISION_APPLE) {
                tp += counts[Sample.APPLE][p];
                fp += counts[Sample.OTHER][p];
            } else {
                fn += counts[Sample.APPLE][p];
                tn += counts[Sample.OTHER][p];
            }
        }
        return new int[]{tp, tn, fp, fn};
    }

    /**
     * Сериализует матрицу в CSV: заголовок с именами классов, строка решений, две строки данных.
     *
     * @return текст CSV
     */
    public String toCsv() {
        StringBuilder sb = new StringBuilder();
        sb.append("true\\predicted");
        for (String n : partition.classes()) {
            sb.append(',').append(n);
        }
        sb.append('\n').append("decision");
        for (int p = 0; p < classCount(); p++) {
            sb.append(',').append(partition.decisionFor(p) == ClassPartition.DECISION_APPLE ? "l1" : "l2");
        }
        sb.append('\n').append("apple");
        for (int c : counts[Sample.APPLE]) {
            sb.append(',').append(c);
        }
        sb.append('\n').append("other");
        for (int c : counts[Sample.OTHER]) {
            sb.append(',').append(c);
        }
        sb.append('\n');
        return sb.toString();
    }

    /**
     * Восстанавливает матрицу из строк CSV, созданных {@link #toCsv()}.
     * Строки после четвёртой (метрики) игнорируются.
     *
     * @param partitionName имя разбиения
     * @param lines         строки файла
     * @return матрица
     * @throws IllegalArgumentException при неверном формате
     */
    public static ConfusionMatrix fromCsv(String partitionName, List<String> lines) {
        if (lines == null || lines.size() < 4) {
            throw new IllegalArgumentException("Матрица ошибок: ожидается не менее 4 строк");
        }
        String[] header = lines.get(0).split(",", -1);
        String[] dec = lines.get(1).split(",", -1);
        String[] apple = lines.get(2).split(",", -1);
        String[] other = lines.get(3).split(",", -1);
        int m = header.length - 1;
        if (m < 1 || dec.length != m + 1 || apple.length != m + 1 || other.length != m + 1) {
            throw new IllegalArgumentException("Матрица ошибок: несогласованное число столбцов");
        }
        List<String> names = new ArrayList<>();
        int[] decisions = new int[m];
        int[] identity = new int[m];
        for (int i = 0; i < m; i++) {
            names.add(header[i + 1].trim());
            decisions[i] = dec[i + 1].trim().equals("l1")
                    ? ClassPartition.DECISION_APPLE : ClassPartition.DECISION_NOT_APPLE;
            identity[i] = i;
        }
        ConfusionMatrix cm = new ConfusionMatrix(new ClassPartition(partitionName, names, decisions, identity));
        try {
            for (int i = 0; i < m; i++) {
                cm.add(Sample.APPLE, i, Integer.parseInt(apple[i + 1].trim()));
                cm.add(Sample.OTHER, i, Integer.parseInt(other[i + 1].trim()));
            }
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Матрица ошибок: нечисловое значение", e);
        }
        return cm;
    }
}
