// src/main/java/volsu/model/Sample.java
package volsu.model;

import java.io.File;

/**
 * Образец обучающей/тестовой выборки: файл изображения, истинная метка,
 * принадлежность к подвыборке (train/test) и (после извлечения) вектор признаков.
 */
public final class Sample {

    /** Метка: на изображении яблоко. */
    public static final int APPLE = 0;
    /** Метка: на изображении не яблоко. */
    public static final int OTHER = 1;

    private final File file;
    private final int label;
    private String split;
    private double[] features;

    /**
     * Создаёт образец.
     *
     * @param file  файл изображения
     * @param label истинная метка ({@link #APPLE} или {@link #OTHER})
     * @param split подвыборка ("train" / "test") либо {@code null}, если ещё не назначена
     */
    public Sample(File file, int label, String split) {
        this.file = file;
        this.label = label;
        this.split = split;
    }

    /**
     * Возвращает файл изображения.
     *
     * @return файл
     */
    public File getFile() {
        return file;
    }

    /**
     * Возвращает истинную метку.
     *
     * @return {@link #APPLE} или {@link #OTHER}
     */
    public int getLabel() {
        return label;
    }

    /**
     * Возвращает подвыборку образца.
     *
     * @return "train", "test" или {@code null}
     */
    public String getSplit() {
        return split;
    }

    /**
     * Назначает подвыборку образца.
     *
     * @param split "train" или "test"
     */
    public void setSplit(String split) {
        this.split = split;
    }

    /**
     * Возвращает вектор признаков (в порядке {@code FeatureExtractor.featureNames()}).
     *
     * @return вектор признаков или {@code null}, если ещё не извлечён
     */
    public double[] getFeatures() {
        return features;
    }

    /**
     * Сохраняет вектор признаков образца.
     *
     * @param features вектор признаков
     */
    public void setFeatures(double[] features) {
        this.features = features;
    }

    @Override
    public String toString() {
        return file + " [" + (label == APPLE ? "apple" : "other") + ", " + split + "]";
    }
}
