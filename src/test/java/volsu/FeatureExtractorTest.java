// src/test/java/volsu/FeatureExtractorTest.java
package volsu;

import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit-тесты признаков {@link FeatureExtractor}.
 */
class FeatureExtractorTest {

    private final FeatureExtractor fx = new FeatureExtractor();

    /**
     * Создаёт однотонное изображение.
     *
     * @param w ширина
     * @param h высота
     * @param c цвет
     * @return изображение
     */
    private static BufferedImage solid(int w, int h, Color c) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(c);
        g.fillRect(0, 0, w, h);
        g.dispose();
        return img;
    }

    /**
     * Создаёт белый холст.
     *
     * @param w ширина
     * @param h высота
     * @return изображение
     */
    private static BufferedImage canvas(int w, int h) {
        return solid(w, h, Color.WHITE);
    }

    /**
     * Рисует эллипс.
     *
     * @param img изображение
     * @param x   левый край
     * @param y   верхний край
     * @param w   ширина
     * @param h   высота
     * @param c   цвет
     */
    private static void oval(BufferedImage img, int x, int y, int w, int h, Color c) {
        Graphics2D g = img.createGraphics();
        g.setColor(c);
        g.fillOval(x, y, w, h);
        g.dispose();
    }

    /**
     * Рисует прямоугольник.
     *
     * @param img изображение
     * @param x   левый край
     * @param y   верхний край
     * @param w   ширина
     * @param h   высота
     * @param c   цвет
     */
    private static void rect(BufferedImage img, int x, int y, int w, int h, Color c) {
        Graphics2D g = img.createGraphics();
        g.setColor(c);
        g.fillRect(x, y, w, h);
        g.dispose();
    }

    /** Проверяет преобразование RGB→HSV на известных цветах. */
    @Test
    void testRgbToHsv() {
        float[] red = fx.rgbToHsv(0xFF0000);
        assertEquals(0f, red[0], 1f);
        assertEquals(1f, red[1], 1e-3);
        assertEquals(1f, red[2], 1e-3);
        assertEquals(120f, fx.rgbToHsv(0x00FF00)[0], 1f);
        assertEquals(60f, fx.rgbToHsv(0xFFFF00)[0], 1f);
        float[] white = fx.rgbToHsv(0xFFFFFF);
        assertEquals(0f, white[1], 1e-3);
        assertEquals(1f, white[2], 1e-3);
        assertEquals(0f, fx.rgbToHsv(0x000000)[2], 1e-3);
    }

    /** Полностью красное изображение даёт долю красного ≈ 1, синее — ≈ 0. */
    @Test
    void testRedRatio() {
        assertTrue(fx.redRatio(solid(64, 64, Color.RED)) > 0.99);
        assertTrue(fx.redRatio(solid(64, 64, Color.BLUE)) < 0.01);
    }

    /** Полностью зелёное изображение даёт долю зелёного ≈ 1. */
    @Test
    void testGreenRatio() {
        assertTrue(fx.greenRatio(solid(64, 64, new Color(0, 255, 0))) > 0.99);
    }

    /** Жёлтое изображение даёт долю жёлтого ≈ 1. */
    @Test
    void testYellowRatio() {
        assertTrue(fx.yellowRatio(solid(64, 64, new Color(255, 255, 0))) > 0.99);
    }

    /** Круг имеет округлость ≈ 1, вытянутый прямоугольник — заметно меньше 0.8. */
    @Test
    void testRoundness() {
        BufferedImage circle = canvas(200, 200);
        oval(circle, 20, 20, 160, 160, Color.RED);
        assertTrue(fx.roundness(circle) > 0.9, "круг: " + fx.roundness(circle));

        BufferedImage bar = canvas(200, 200);
        rect(bar, 20, 85, 160, 30, Color.RED);
        assertTrue(fx.roundness(bar) < 0.8, "прямоугольник: " + fx.roundness(bar));
    }

    /** Квадрат даёт соотношение сторон 1.0, широкий прямоугольник — больше 1. */
    @Test
    void testAspectRatio() {
        BufferedImage square = canvas(200, 200);
        rect(square, 50, 50, 100, 100, Color.RED);
        assertEquals(1.0, fx.aspectRatio(square), 0.05);

        BufferedImage wide = canvas(200, 200);
        rect(wide, 20, 75, 150, 50, Color.RED);
        assertTrue(fx.aspectRatio(wide) > 1.0);
    }

    /** Длина вектора равна числу признаков словаря. */
    @Test
    void testFeatureVectorLength() {
        double[] f = fx.extract(solid(50, 50, Color.RED));
        assertEquals(FeatureExtractor.featureNames().size(), f.length);
    }

    /** Извлечение признаков для изображения 1024×1024 укладывается в 1 с. */
    @Test
    void testExtractionTime() {
        BufferedImage big = canvas(1024, 1024);
        oval(big, 200, 200, 600, 600, Color.RED);
        fx.extract(canvas(64, 64));
        long t0 = System.nanoTime();
        fx.extract(big);
        long ms = (System.nanoTime() - t0) / 1_000_000;
        assertTrue(ms <= 1000, "время, мс: " + ms);
    }
}
