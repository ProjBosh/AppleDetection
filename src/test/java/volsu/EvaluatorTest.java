// src/test/java/volsu/EvaluatorTest.java
package volsu;

import org.junit.jupiter.api.Test;
import volsu.model.ClassPartition;
import volsu.model.ConfusionMatrix;
import volsu.model.ModelConfig;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Тесты метрик {@link Evaluator}.
 */
class EvaluatorTest {

    private static final int[] TRUTH = {0, 0, 0, 1, 1, 1, 1};
    private static final int[] PRED = {0, 0, 1, 1, 2, 0, 1};

    /** Матрица ошибок на заранее известных предсказаниях. */
    @Test
    void testConfusionMatrix() {
        ConfusionMatrix cm = Evaluator.confusionMatrix(ClassPartition.A1, TRUTH, PRED);
        assertEquals(2, cm.get(0, 0));
        assertEquals(1, cm.get(0, 1));
        assertEquals(0, cm.get(0, 2));
        assertEquals(1, cm.get(1, 0));
        assertEquals(2, cm.get(1, 1));
        assertEquals(1, cm.get(1, 2));
        assertEquals(7, cm.total());
        assertArrayEquals(new int[]{2, 3, 1, 1}, Evaluator.binaryConfusionMatrix(ClassPartition.A1, TRUTH, PRED));
    }

    /** Accuracy = (2 + 3) / 7. */
    @Test
    void testAccuracy() {
        assertEquals(5.0 / 7.0, Evaluator.accuracy(2, 3, 1, 1), 1e-12);
    }

    /** Precision = 2 / (2 + 1). */
    @Test
    void testPrecision() {
        assertEquals(2.0 / 3.0, Evaluator.precision(2, 1), 1e-12);
    }

    /** Recall = 2 / (2 + 1). */
    @Test
    void testRecall() {
        assertEquals(2.0 / 3.0, Evaluator.recall(2, 1), 1e-12);
    }

    /** Пустая выборка не приводит к делению на ноль. */
    @Test
    void testEmptyTestSet() throws Exception {
        assertEquals(0.0, Evaluator.accuracy(0, 0, 0, 0));
        assertEquals(0.0, Evaluator.precision(0, 0));
        assertEquals(0.0, Evaluator.recall(0, 0));
        Map<String, double[]> th = new LinkedHashMap<>();
        th.put("roundness", new double[]{0.7, 1.0, 0.4, 1.0});
        Classifier clf = new Classifier(new ModelConfig(ClassPartition.A1, List.of("roundness"), th, 256, 256));
        Evaluator.EvaluationResult r = new Evaluator().evaluate(clf, new ArrayList<>());
        assertEquals(0.0, r.getAccuracy());
        assertEquals(0.0, r.getPrecision());
        assertEquals(0.0, r.getRecall());
    }

    /** Формат CSV: заголовок, строка решений, две строки данных; обратное чтение даёт ту же матрицу. */
    @Test
    void testCsvOutput() {
        ConfusionMatrix cm = Evaluator.confusionMatrix(ClassPartition.A1, TRUTH, PRED);
        String[] lines = cm.toCsv().split("\n");
        assertEquals(4, lines.length);
        assertTrue(lines[0].startsWith("true\\predicted"));
        assertEquals("decision,l1,l2,l2", lines[1]);
        assertEquals("apple,2,1,0", lines[2]);
        assertEquals("other,1,2,1", lines[3]);
        ConfusionMatrix back = ConfusionMatrix.fromCsv("A1", List.of(lines));
        assertArrayEquals(cm.binary(), back.binary());
        assertEquals(cm.total(), back.total());
    }
}
