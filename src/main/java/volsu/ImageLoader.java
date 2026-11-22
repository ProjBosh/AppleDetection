// src/main/java/volsu/ImageLoader.java
package volsu;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Загрузка и предобработка изображения: чтение через {@link ImageIO}, приведение к
 * {@link BufferedImage#TYPE_INT_RGB}, масштабирование до единого размера и (опционально)
 * удаление фона.
 * <p>
 * Ограничения ресурсов C₀: размер входа ≤ {@value #MAX_SIZE}×{@value #MAX_SIZE},
 * время обработки ≤ 1 с (контролируется в {@code Main}).
 */
public class ImageLoader {

    /** Максимальный размер стороны входного изображения (ограничение C₀). */
    public static final int MAX_SIZE = 1024;

    /**
     * Загружает изображение из файла. Слишком большие изображения (> 1024 по любой стороне)
     * сразу уменьшаются, чтобы уложиться в ограничения памяти и времени.
     *
     * @param file файл изображения
     * @return загруженное изображение
     * @throws ImageNotFoundException если файл не существует или не является файлом
     * @throws NotAnImageException    если файл не является читаемым изображением
     */
    public BufferedImage load(File file) throws ImageNotFoundException, NotAnImageException {
        if (file == null || !file.exists() || !file.isFile()) {
            throw new ImageNotFoundException("Файл не найден: " + file);
        }
        BufferedImage img;
        try {
            img = ImageIO.read(file);
        } catch (IOException | RuntimeException e) {
            throw new NotAnImageException("Не удалось прочитать изображение: " + file, e);
        }
        if (img == null) {
            throw new NotAnImageException("Файл не является изображением: " + file);
        }
        if (img.getWidth() > MAX_SIZE || img.getHeight() > MAX_SIZE) {
            img = resize(img, MAX_SIZE, MAX_SIZE);
        }
        return img;
    }

    /**
     * Полный конвейер предобработки: загрузка → приведение к RGB → масштабирование.
     * Одинаков для классификации, обучения и оценки.
     *
     * @param file файл изображения
     * @param w    максимальная ширина
     * @param h    максимальная высота
     * @return подготовленное изображение
     * @throws ImageNotFoundException если файл не найден
     * @throws NotAnImageException    если файл не изображение
     */
    public BufferedImage loadPrepared(File file, int w, int h)
            throws ImageNotFoundException, NotAnImageException {
        return resize(toRgb(load(file)), w, h);
    }

    /**
     * Масштабирует изображение так, чтобы оно вписывалось в {@code w}×{@code h} с сохранением
     * пропорций: {@code scale = min(w / W, h / H)}. При сильном уменьшении используется
     * последовательное уменьшение вдвое (качественнее, чем одно билинейное преобразование).
     *
     * @param img исходное изображение
     * @param w   максимальная ширина
     * @param h   максимальная высота
     * @return масштабированное изображение (или исходное, если размер совпал)
     */
    public BufferedImage resize(BufferedImage img, int w, int h) {
        double scale = Math.min((double) w / img.getWidth(), (double) h / img.getHeight());
        int tw = Math.max(1, (int) Math.round(img.getWidth() * scale));
        int th = Math.max(1, (int) Math.round(img.getHeight() * scale));
        if (tw == img.getWidth() && th == img.getHeight()) {
            return img;
        }
        BufferedImage cur = img;
        while (cur.getWidth() / 2 >= tw && cur.getHeight() / 2 >= th) {
            cur = scaleOnce(cur, cur.getWidth() / 2, cur.getHeight() / 2);
        }
        return scaleOnce(cur, tw, th);
    }

    /**
     * Однократное билинейное масштабирование на белом фоне (прозрачность композируется на белое).
     *
     * @param src исходное изображение
     * @param w   новая ширина
     * @param h   новая высота
     * @return новое изображение типа TYPE_INT_RGB
     */
    private BufferedImage scaleOnce(BufferedImage src, int w, int h) {
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g.drawImage(src, 0, 0, w, h, Color.WHITE, null);
        g.dispose();
        return out;
    }

    /**
     * Приводит изображение к единому типу {@link BufferedImage#TYPE_INT_RGB}
     * (прозрачные области заливаются белым).
     *
     * @param img исходное изображение
     * @return изображение типа TYPE_INT_RGB
     */
    public BufferedImage toRgb(BufferedImage img) {
        if (img.getType() == BufferedImage.TYPE_INT_RGB) {
            return img;
        }
        BufferedImage out = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        g.drawImage(img, 0, 0, Color.WHITE, null);
        g.dispose();
        return out;
    }

    /**
     * Удаляет (маскирует) фон: пиксели вне маски объекта заливаются белым. Маска строится
     * методом {@link FeatureExtractor#segmentObject(BufferedImage)} (отличие от цвета фона по краям).
     *
     * @param img исходное изображение
     * @return новое изображение с белым фоном
     */
    public BufferedImage removeBackground(BufferedImage img) {
        boolean[] mask = new FeatureExtractor().segmentObject(img);
        int w = img.getWidth();
        int h = img.getHeight();
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                out.setRGB(x, y, mask[y * w + x] ? img.getRGB(x, y) : 0xFFFFFF);
            }
        }
        return out;
    }
}
