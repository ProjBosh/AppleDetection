// src/test/java/volsu/ClassifierTest.java
package volsu;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import volsu.Classifier.ClassificationResult;
import volsu.model.ClassPartition;
import volsu.model.ModelConfig;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Тесты решающего правила {@link Classifier}.
 */
class ClassifierTest {

    /**
     * Строит тестовую модель из трёх признаков.
     *
     * @return модель разбиения A1
     */
    private static ModelConfig model() {
        Map<String, double[]> th = new LinkedHashMap<>();
        th.put("colorDominance", new double[]{0.6, 1.0, 0.3, 1.0});
        th.put("roundness", new double[]{0.7, 1.0, 0.4, 1.0});
        th.put("elongation", new double[]{1.0, 1.3, 1.0, 2.0});
        return new ModelConfig(ClassPartition.A1, List.of("colorDominance", "roundness", "elongation"),
                th, 256, 256);
    }

    /**
     * Формирует полный вектор признаков с тремя заданными значениями.
     *
     * @param dom   colorDominance
     * @param round roundness
     * @param elong elongation
     * @return вектор
     */
    private static double[] vec(double dom, double round, double elong) {
        List<String> names = FeatureExtractor.featureNames();
        double[] f = new double[names.size()];
        f[names.indexOf("colorDominance")] = dom;
        f[names.indexOf("roundness")] = round;
        f[names.indexOf("elongation")] = elong;
        return f;
    }

    /** Типичные признаки яблока → решение «яблоко», класс ΩA1_1. */
    @Test
    void testAppleClass() throws Exception {
        Classifier clf = new Classifier(model());
        ClassificationResult r = clf.classify(vec(0.9, 0.9, 1.1));
        assertEquals(0, r.getClassIndex());
        assertTrue(r.isApple());
        assertTrue(r.getConfidence() >= 0.6);
    }

    /** Явно не яблоко → решение «не яблоко», класс ΩA1_2. */
    @Test
    void testNotAppleClass() throws Exception {
        Classifier clf = new Classifier(model());
        ClassificationResult r = clf.classify(vec(0.05, 0.9, 1.1));
        assertEquals(1, r.getClassIndex());
        assertFalse(r.isApple());
    }

    /** Класс ΩA1_3 → «не яблоко» с низкой уверенностью. */
    @Test
    void testUnsureClass() throws Exception {
        Classifier clf = new Classifier(model());
        ClassificationResult r = clf.classify(vec(0.45, 0.9, 1.1));
        assertEquals(2, r.getClassIndex());
        assertFalse(r.isApple());
        assertTrue(r.getConfidence() <= 0.5);
    }

    /** Уверенность всегда лежит в диапазоне [0, 1]. */
    @Test
    void testConfidenceRange() throws Exception {
        Classifier clf = new Classifier(model());
        double[][] cases = {vec(0.9, 0.9, 1.1), vec(0.0, 0.0, 5.0), vec(0.45, 0.5, 1.5), vec(1.0, 1.0, 1.0)};
        for (double[] c : cases) {
            double conf = clf.confidence(c);
            assertTrue(conf >= 0.0 && conf <= 1.0, "уверенность: " + conf);
        }
    }

    /** Соответствие классов решениям L для A1 и для объединённого A2. */
    @Test
    void testDecisionMapping() {
        ClassPartition a1 = ClassPartition.A1;
        assertEquals(ClassPartition.DECISION_APPLE, a1.decisionFor(0));
        assertEquals(ClassPartition.DECISION_NOT_APPLE, a1.decisionFor(1));
        assertEquals(ClassPartition.DECISION_NOT_APPLE, a1.decisionFor(2));
        ClassPartition a2 = a1.merge(2, 1);
        assertEquals(2, a2.size());
        assertArrayEquals(new int[]{0, 1, 1}, a2.getBaseToLocal());
        assertEquals("A2", a2.name());
    }

    /** Модель сохраняется в JSON и загружается без потерь. */
    @Test
    void testModelLoad(@TempDir Path tmp) throws Exception {
        ModelConfig cfg = model();
        Path file = tmp.resolve("model.json");
        cfg.save(file);
        ModelConfig loaded = ModelConfig.load(file);
        assertEquals(cfg.getFeatures(), loaded.getFeatures());
        assertArrayEquals(cfg.getThresholds().get("roundness"), loaded.getThresholds().get("roundness"), 1e-12);
        assertEquals(cfg.getPartition().name(), loaded.getPartition().name());
        assertTrue(Classifier.load(file.toString()).classify(vec(0.9, 0.9, 1.1)).isApple());
    }

    /** Повреждённая модель вызывает {@link ModelException} (код 2). */
    @Test
    void testBrokenModel(@TempDir Path tmp) throws Exception {
        Path file = tmp.resolve("broken.json");
        Files.writeString(file, "{ not a json");
        assertThrows(ModelException.class, () -> ModelConfig.load(file));
        Path file2 = tmp.resolve("broken2.json");
        Files.writeString(file2, "{\"version\": 1}");
        assertThrows(ModelException.class, () -> ModelConfig.load(file2));
    }
}
