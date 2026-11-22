// src/test/java/volsu/EdgeCasesTest.java
package volsu;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import volsu.Classifier.ClassificationResult;
import volsu.model.ClassPartition;
import volsu.model.ModelConfig;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Граничные случаи: зелёное яблоко, помидор, красный шар, слабый свет, несколько объектов,
 * ошибки ввода, большие изображения.
 */
class EdgeCasesTest {

    /**
     * Строит модель с признаками colorDominance, roundness, elongation, meanSaturation.
     *
     * @return модель
     */
    private static ModelConfig model() {
        Map<String, double[]> th = new LinkedHashMap<>();
        th.put("colorDominance", new double[]{0.6, 1.0, 0.3, 1.0});
        th.put("roundness", new double[]{0.8, 1.0, 0.4, 1.0});
        th.put("elongation", new double[]{1.0, 1.25, 1.0, 2.0});
        th.put("meanSaturation", new double[]{0.35, 0.85, 0.2, 1.0});
        return new ModelConfig(ClassPartition.A1,
                List.of("colorDominance", "roundness", "elongation", "meanSaturation"), th, 256, 256);
    }

    /**
     * Создаёт белый холст.
     *
     * @param w ширина
     * @param h высота
     * @return изображение
     */
    private static BufferedImage canvas(int w, int h) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, w, h);
        g.dispose();
        return img;
    }

    /**
     * Рисует эллипс на изображении.
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
     * Классифицирует изображение моделью {@link #model()}.
     *
     * @param img изображение
     * @return результат
     * @throws Exception при ошибке модели
     */
    private static ClassificationResult classify(BufferedImage img) throws Exception {
        Classifier clf = new Classifier(model());
        return clf.classify(new FeatureExtractor().extract(img));
    }

    /**
     * Записывает модель во временный файл.
     *
     * @param dir каталог
     * @return путь к файлу модели
     * @throws Exception при ошибке записи
     */
    private static Path writeModel(Path dir) throws Exception {
        Path p = dir.resolve("model.json");
        model().save(p);
        return p;
    }

    /** Зелёное яблоко распознаётся как яблоко. */
    @Test
    void testGreenApple() throws Exception {
        BufferedImage img = canvas(200, 200);
        oval(img, 40, 40, 120, 120, new Color(140, 200, 60));
        ClassificationResult r = classify(img);
        assertTrue(r.isApple());
    }

    /** Приплюснутый помидор — не яблоко (класс «похоже на яблоко»). */
    @Test
    void testTomato() throws Exception {
        BufferedImage img = canvas(200, 200);
        oval(img, 30, 50, 140, 100, new Color(200, 40, 40));
        ClassificationResult r = classify(img);
        assertFalse(r.isApple());
    }

    /** Красный шар (насыщенность 1.0) → класс ΩA1_3, «не яблоко» с низкой уверенностью. */
    @Test
    void testRedBall() throws Exception {
        BufferedImage img = canvas(200, 200);
        oval(img, 40, 40, 120, 120, new Color(255, 0, 0));
        ClassificationResult r = classify(img);
        assertEquals(2, r.getClassIndex());
        assertFalse(r.isApple());
        assertTrue(r.getConfidence() <= 0.5);
    }

    /** Слабое освещение: обработка без исключений, уверенность в допустимых пределах. */
    @Test
    void testLowLight() throws Exception {
        BufferedImage img = canvas(200, 200);
        oval(img, 40, 40, 120, 120, new Color(40, 8, 8));
        ClassificationResult r = classify(img);
        assertTrue(r.getConfidence() >= 0.0 && r.getConfidence() <= 1.0);
        assertFalse(r.isApple());
    }

    /** Несколько объектов: решение принимается по наибольшему (доминирующему) объекту. */
    @Test
    void testMultipleObjects() throws Exception {
        BufferedImage img = canvas(280, 200);
        oval(img, 20, 40, 120, 120, new Color(200, 40, 40));
        oval(img, 200, 80, 40, 40, new Color(10, 10, 200));
        ClassificationResult r = classify(img);
        assertTrue(r.isApple());
    }

    /** Несуществующий файл изображения → код возврата 1. */
    @Test
    void testFileNotFound(@TempDir Path tmp) throws Exception {
        Path model = writeModel(tmp);
        Path missing = tmp.resolve("no_such_file.jpg");
        int code = Main.parseCommands("detect --model \"" + model + "\" \"" + missing + "\"");
        assertEquals(1, code);
    }

    /** Файл, не являющийся изображением → код возврата 1. */
    @Test
    void testNotAnImage(@TempDir Path tmp) throws Exception {
        Path model = writeModel(tmp);
        Path fake = tmp.resolve("hello.jpg");
        Files.writeString(fake, "this is not an image");
        int code = Main.parseCommands("detect --model \"" + model + "\" \"" + fake + "\"");
        assertEquals(1, code);
    }

    /** Изображение больше 1024×1024 уменьшается при загрузке и успешно обрабатывается. */
    @Test
    void testHugeImage(@TempDir Path tmp) throws Exception {
        BufferedImage img = canvas(1500, 1500);
        oval(img, 300, 300, 900, 900, new Color(200, 40, 40));
        Path file = tmp.resolve("huge.png");
        ImageIO.write(img, "png", file.toFile());
        BufferedImage loaded = new ImageLoader().load(file.toFile());
        assertTrue(loaded.getWidth() <= ImageLoader.MAX_SIZE && loaded.getHeight() <= ImageLoader.MAX_SIZE);
        Path model = writeModel(tmp);
        assertEquals(0, Main.parseCommands("detect --model \"" + model + "\" \"" + file + "\""));
    }
}
