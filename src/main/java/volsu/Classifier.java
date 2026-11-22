// src/main/java/volsu/Classifier.java
package volsu;

import volsu.model.ClassPartition;
import volsu.model.ModelConfig;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Решающее правило и вывод класса с уверенностью.
 * <p>
 * Правило для базового разбиения A1 (для каждого признака модели заданы строгий интервал
 * {@code [sLo, sHi]} и широкий {@code [lLo, lHi]}):
 * <ul>
 *   <li>все признаки в строгих интервалах → ΩA1_1 «яблоко»;</li>
 *   <li>хотя бы один признак вне широкого интервала → ΩA1_2 «явно не яблоко»;</li>
 *   <li>иначе → ΩA1_3 «похоже на яблоко».</li>
 * </ul>
 * Для остальных разбиений класс A1 переводится в класс разбиения через
 * {@link ClassPartition#baseToLocal(int)}, затем в решение L.
 * <p>
 * Уверенность (0..1):
 * <ul>
 *   <li>ΩA1_1: {@code 0.6 + 0.4·m}, где m — минимальный нормированный запас до границы строгого интервала;</li>
 *   <li>ΩA1_2: {@code 0.6 + 0.4·min(1, 4·d)}, d — максимальный выход за широкий интервал
 *       (в долях его ширины);</li>
 *   <li>ΩA1_3: {@code uMax − (uMax − uMin)·min(1, 4·d)}, d — максимальный выход за строгий интервал.</li>
 * </ul>
 */
public final class Classifier {

    private static final double EPS = 1e-9;

    private final ModelConfig config;
    private final int[] featureIdx;
    private final double[][] limits;

    /**
     * Результат классификации.
     */
    public static final class ClassificationResult {
        private final int decision;
        private final int classIndex;
        private final double confidence;
        private final double[] featureVector;

        /**
         * Создаёт результат.
         *
         * @param decision      решение (0 — l1 «яблоко», 1 — l2 «не яблоко»)
         * @param classIndex    индекс класса разбиения модели
         * @param confidence    уверенность 0..1
         * @param featureVector исходный вектор признаков
         */
        public ClassificationResult(int decision, int classIndex, double confidence, double[] featureVector) {
            this.decision = decision;
            this.classIndex = classIndex;
            this.confidence = confidence;
            this.featureVector = featureVector;
        }

        /**
         * Решение из L.
         *
         * @return {@link ClassPartition#DECISION_APPLE} или {@link ClassPartition#DECISION_NOT_APPLE}
         */
        public int getDecision() {
            return decision;
        }

        /**
         * Индекс класса в разбиении модели.
         *
         * @return индекс
         */
        public int getClassIndex() {
            return classIndex;
        }

        /**
         * Уверенность.
         *
         * @return значение 0..1
         */
        public double getConfidence() {
            return confidence;
        }

        /**
         * Вектор признаков.
         *
         * @return вектор
         */
        public double[] getFeatureVector() {
            return featureVector;
        }

        /**
         * Признак итогового решения «яблоко».
         *
         * @return {@code true}, если решение l1
         */
        public boolean isApple() {
            return decision == ClassPartition.DECISION_APPLE;
        }
    }

    /** Внутренний результат анализа: базовый класс A1 и уверенность. */
    private static final class Verdict {
        int baseClass;
        double confidence;
    }

    /**
     * Создаёт классификатор по модели.
     *
     * @param config модель (проверяется {@link ModelConfig#validate()})
     * @throws ModelException если модель некорректна (→ код возврата 2)
     */
    public Classifier(ModelConfig config) throws ModelException {
        if (config == null) {
            throw new ModelException("Модель не задана");
        }
        config.validate();
        this.config = config;
        List<String> all = FeatureExtractor.featureNames();
        List<String> names = config.getFeatures();
        this.featureIdx = new int[names.size()];
        this.limits = new double[names.size()][];
        Map<String, double[]> th = config.getThresholds();
        for (int k = 0; k < names.size(); k++) {
            featureIdx[k] = all.indexOf(names.get(k));
            limits[k] = th.get(names.get(k)).clone();
        }
    }

    /**
     * Загружает классификатор из файла модели {@code model_A*.json}.
     *
     * @param modelPath путь к JSON
     * @return классификатор
     * @throws ModelException при ошибке чтения/валидации (→ код 2)
     */
    public static Classifier load(String modelPath) throws ModelException {
        return new Classifier(ModelConfig.load(Path.of(modelPath)));
    }

    /**
     * Конфигурация модели.
     *
     * @return модель
     */
    public ModelConfig getConfig() {
        return config;
    }

    /**
     * Классифицирует объект: класс → решение L → уверенность.
     *
     * @param features полный вектор признаков (порядок {@link FeatureExtractor#featureNames()})
     * @return результат классификации
     * @throws IllegalArgumentException при неверной длине вектора
     */
    public ClassificationResult classify(double[] features) {
        Verdict v = analyse(features);
        int local = config.getPartition().baseToLocal(v.baseClass);
        int decision = mapToDecision(local);
        return new ClassificationResult(decision, local, v.confidence, features);
    }

    /**
     * Определяет класс разбиения модели (ΩA1_1..3 либо классы A2 после объединения).
     *
     * @param features вектор признаков
     * @return индекс класса разбиения
     */
    public int classifyClass(double[] features) {
        return config.getPartition().baseToLocal(applyThresholds(features));
    }

    /**
     * Переводит класс разбиения в решение из L.
     *
     * @param classIndex индекс класса
     * @return 0 — l1 «яблоко», 1 — l2 «не яблоко»
     */
    public int mapToDecision(int classIndex) {
        return config.getPartition().decisionFor(classIndex);
    }

    /**
     * Рассчитывает уверенность решения (см. описание класса).
     *
     * @param features вектор признаков
     * @return уверенность 0..1
     */
    public double confidence(double[] features) {
        return analyse(features).confidence;
    }

    /**
     * Применяет решающие границы к признакам.
     *
     * @param features вектор признаков
     * @return индекс класса базового разбиения A1: 0 — яблоко, 1 — явно не яблоко, 2 — похоже на яблоко
     */
    public int applyThresholds(double[] features) {
        return analyse(features).baseClass;
    }

    /**
     * Единый анализ вектора признаков: класс A1 и уверенность.
     *
     * @param f полный вектор признаков
     * @return вердикт
     */
    private Verdict analyse(double[] f) {
        if (f == null || f.length != FeatureExtractor.featureNames().size()) {
            throw new IllegalArgumentException("Неверная длина вектора признаков");
        }
        boolean allStrict = true;
        boolean rejected = false;
        double minInner = 1.0;
        double maxOutStrict = 0.0;
        double maxOutLoose = 0.0;
        for (int k = 0; k < featureIdx.length; k++) {
            double x = f[featureIdx[k]];
            double[] t = limits[k];
            double sLo = t[0];
            double sHi = t[1];
            double lLo = t[2];
            double lHi = t[3];
            double lWidth = Math.max(lHi - lLo, EPS);
            double sWidth = Math.max(sHi - sLo, EPS);
            if (x < sLo || x > sHi) {
                allStrict = false;
                double d = (x < sLo ? sLo - x : x - sHi) / lWidth;
                maxOutStrict = Math.max(maxOutStrict, d);
            } else {
                double m = Math.min(x - sLo, sHi - x) / (0.5 * sWidth);
                minInner = Math.min(minInner, Math.min(1.0, m));
            }
            if (x < lLo || x > lHi) {
                rejected = true;
                double d = (x < lLo ? lLo - x : x - lHi) / lWidth;
                maxOutLoose = Math.max(maxOutLoose, d);
            }
        }
        Verdict v = new Verdict();
        if (allStrict) {
            v.baseClass = 0;
            v.confidence = 0.6 + 0.4 * minInner;
        } else if (rejected) {
            v.baseClass = 1;
            v.confidence = 0.6 + 0.4 * Math.min(1.0, 4.0 * maxOutLoose);
        } else {
            v.baseClass = 2;
            double lo = config.getUnsureMinConfidence();
            double hi = config.getUnsureMaxConfidence();
            v.confidence = hi - (hi - lo) * Math.min(1.0, 4.0 * maxOutStrict);
        }
        v.confidence = Math.max(0.0, Math.min(1.0, v.confidence));
        return v;
    }
}
