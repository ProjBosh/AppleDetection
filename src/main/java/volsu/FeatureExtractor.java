// src/main/java/volsu/FeatureExtractor.java
package volsu;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Извлечение признаков априорного словаря. Все признаки вычисляются по маске объекта
 * (см. {@link #segmentObject(BufferedImage)}), а не по всему кадру.
 * <p>
 * Порядок элементов вектора задаётся {@link #featureNames()}:
 * <ol>
 *   <li>{@code redRatio}, {@code greenRatio}, {@code yellowRatio} — доли красных/зелёных/жёлтых пикселей;</li>
 *   <li>{@code colorDominance} = max(red, green, yellow) — доля доминирующего «яблочного» цвета;</li>
 *   <li>{@code meanHue} (градусы, круговое среднее), {@code meanSaturation}, {@code meanValue};</li>
 *   <li>{@code roundness} = A / (π · r<sub>max</sub>²) — округлость, 1 для круга;</li>
 *   <li>{@code aspectRatio} = ширина / высота ограничивающего прямоугольника;</li>
 *   <li>{@code elongation} = max(aspectRatio, 1/aspectRatio) ≥ 1;</li>
 *   <li>{@code texture} — средняя амплитуда градиента яркости (0..1);</li>
 *   <li>{@code brightnessHist0..7} — нормированная гистограмма яркости V (8 интервалов).</li>
 * </ol>
 * Для уложения в ограничение C₀ (≤ 1 с) изображения больше {@value #ANALYSIS_SIZE} пикселей
 * по стороне предварительно уменьшаются.
 */
public class FeatureExtractor {

    /** Число интервалов гистограммы яркости. */
    public static final int HIST_BINS = 8;
    /** Максимальный размер стороны при анализе. */
    public static final int ANALYSIS_SIZE = 256;

    private static final List<String> NAMES;
    private static final int IDX_RED = 0;
    private static final int IDX_GREEN = 1;
    private static final int IDX_YELLOW = 2;
    private static final int IDX_DOMINANCE = 3;
    private static final int IDX_HUE = 4;
    private static final int IDX_SAT = 5;
    private static final int IDX_VAL = 6;
    private static final int IDX_ROUND = 7;
    private static final int IDX_ASPECT = 8;
    private static final int IDX_ELONG = 9;
    private static final int IDX_TEXTURE = 10;
    private static final int IDX_HIST = 11;

    static {
        List<String> n = new ArrayList<>(List.of("redRatio", "greenRatio", "yellowRatio", "colorDominance",
                "meanHue", "meanSaturation", "meanValue", "roundness", "aspectRatio", "elongation", "texture"));
        for (int i = 0; i < HIST_BINS; i++) {
            n.add("brightnessHist" + i);
        }
        NAMES = Collections.unmodifiableList(n);
    }

    private final ImageLoader loader = new ImageLoader();

    /**
     * Имена признаков в порядке вектора {@link #extract(BufferedImage)}.
     *
     * @return неизменяемый список имён
     */
    public static List<String> featureNames() {
        return NAMES;
    }

    /**
     * Вычисляет полный вектор признаков изображения.
     *
     * @param img изображение
     * @return вектор в порядке {@link #featureNames()}
     * @throws IllegalArgumentException если изображение {@code null}
     */
    public double[] extract(BufferedImage img) {
        if (img == null) {
            throw new IllegalArgumentException("Изображение не задано");
        }
        BufferedImage s = img;
        if (Math.max(s.getWidth(), s.getHeight()) > ANALYSIS_SIZE) {
            s = loader.resize(s, ANALYSIS_SIZE, ANALYSIS_SIZE);
        }
        int w = s.getWidth();
        int h = s.getHeight();
        int[] rgb = s.getRGB(0, 0, w, h, null, 0, w);
        boolean[] mask = segment(rgb, w, h);
        return compute(rgb, w, h, mask);
    }

    /**
     * Преобразует пиксель RGB в HSV.
     * Формулы: {@code V = max}, {@code S = (max − min) / max},
     * {@code H = 60° · ((G − B)/Δ mod 6)} при max = R; {@code 60° · ((B − R)/Δ + 2)} при max = G;
     * {@code 60° · ((R − G)/Δ + 4)} при max = B, где Δ = max − min.
     *
     * @param rgb пиксель в формате 0xRRGGBB
     * @return массив {H (0..360), S (0..1), V (0..1)}
     */
    public float[] rgbToHsv(int rgb) {
        float r = ((rgb >> 16) & 255) / 255f;
        float g = ((rgb >> 8) & 255) / 255f;
        float b = (rgb & 255) / 255f;
        float max = Math.max(r, Math.max(g, b));
        float min = Math.min(r, Math.min(g, b));
        float d = max - min;
        float hue;
        if (d == 0f) {
            hue = 0f;
        } else if (max == r) {
            hue = 60f * (((g - b) / d) % 6f);
        } else if (max == g) {
            hue = 60f * ((b - r) / d + 2f);
        } else {
            hue = 60f * ((r - g) / d + 4f);
        }
        if (hue < 0f) {
            hue += 360f;
        }
        float sat = max == 0f ? 0f : d / max;
        return new float[]{hue, sat, max};
    }

    /**
     * Доля красных пикселей объекта: H ∈ [0°, 20°] ∪ [340°, 360°], S &gt; 0.4, V &gt; 0.2.
     *
     * @param img изображение
     * @return доля 0..1
     */
    public double redRatio(BufferedImage img) {
        return extract(img)[IDX_RED];
    }

    /**
     * Доля зелёных пикселей объекта: H ∈ [70°, 165°], S &gt; 0.3, V &gt; 0.2.
     *
     * @param img изображение
     * @return доля 0..1
     */
    public double greenRatio(BufferedImage img) {
        return extract(img)[IDX_GREEN];
    }

    /**
     * Доля жёлтых пикселей объекта: H ∈ [40°, 70°), S &gt; 0.4, V &gt; 0.3.
     *
     * @param img изображение
     * @return доля 0..1
     */
    public double yellowRatio(BufferedImage img) {
        return extract(img)[IDX_YELLOW];
    }

    /**
     * Средний тон по объекту (круговое среднее: {@code atan2(Σ sin H, Σ cos H)}).
     *
     * @param img изображение
     * @return тон в градусах [0, 360)
     */
    public double meanHue(BufferedImage img) {
        return extract(img)[IDX_HUE];
    }

    /**
     * Средняя насыщенность объекта.
     *
     * @param img изображение
     * @return значение 0..1
     */
    public double meanSaturation(BufferedImage img) {
        return extract(img)[IDX_SAT];
    }

    /**
     * Средняя яркость (V) объекта.
     *
     * @param img изображение
     * @return значение 0..1
     */
    public double meanValue(BufferedImage img) {
        return extract(img)[IDX_VAL];
    }

    /**
     * Округлость контура: {@code roundness = A / (π · (r_max + 0.5)²)}, где A — площадь маски,
     * r_max — максимальное расстояние от центра масс до граничного пикселя.
     * Для круга ≈ 1, для квадрата ≈ 0.64, для вытянутого прямоугольника — существенно меньше.
     *
     * @param img изображение
     * @return значение 0..1
     */
    public double roundness(BufferedImage img) {
        return extract(img)[IDX_ROUND];
    }

    /**
     * Соотношение сторон ограничивающего прямоугольника объекта: ширина / высота.
     *
     * @param img изображение
     * @return значение &gt; 0 (1.0 для квадрата)
     */
    public double aspectRatio(BufferedImage img) {
        return extract(img)[IDX_ASPECT];
    }

    /**
     * Гистограмма яркости объекта: доли пикселей в {@value #HIST_BINS} равных интервалах V.
     *
     * @param img изображение
     * @return массив из {@value #HIST_BINS} долей (сумма = 1)
     */
    public double[] brightnessHistogram(BufferedImage img) {
        double[] f = extract(img);
        double[] h = new double[HIST_BINS];
        System.arraycopy(f, IDX_HIST, h, 0, HIST_BINS);
        return h;
    }

    /**
     * Текстура: средняя амплитуда градиента яркости по объекту,
     * {@code (|g(x+1,y) − g(x−1,y)| + |g(x,y+1) − g(x,y−1)|) / (2·255)} (0..1).
     * Вычисляется за один проход, поэтому укладывается в C₀.
     *
     * @param img изображение
     * @return значение 0..1
     */
    public double texture(BufferedImage img) {
        return extract(img)[IDX_TEXTURE];
    }

    /**
     * Выделяет маску объекта: пиксель считается объектом, если его RGB-расстояние до среднего
     * цвета рамки изображения больше порога {@code max(35, 3·d̄)}, где d̄ — среднее расстояние
     * пикселей рамки до их среднего цвета. Затем берётся наибольшая связная компонента
     * (несколько объектов → доминирующий), заполняются «дыры» (блики). Если объект не найден
     * (площадь &lt; 1 %), маска покрывает весь кадр.
     *
     * @param img изображение
     * @return плоская маска размера {@code width·height} (по строкам)
     */
    public boolean[] segmentObject(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        return segment(img.getRGB(0, 0, w, h, null, 0, w), w, h);
    }

    /**
     * Сегментация объекта по массиву пикселей (см. {@link #segmentObject(BufferedImage)}).
     *
     * @param rgb пиксели
     * @param w   ширина
     * @param h   высота
     * @return маска объекта
     */
    private static boolean[] segment(int[] rgb, int w, int h) {
        int n = w * h;
        int t = Math.max(1, Math.min(w, h) / 32);
        double sr = 0, sg = 0, sb = 0;
        int cnt = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (x < t || y < t || x >= w - t || y >= h - t) {
                    int c = rgb[y * w + x];
                    sr += (c >> 16) & 255;
                    sg += (c >> 8) & 255;
                    sb += c & 255;
                    cnt++;
                }
            }
        }
        sr /= cnt;
        sg /= cnt;
        sb /= cnt;
        double md = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (x < t || y < t || x >= w - t || y >= h - t) {
                    md += dist(rgb[y * w + x], sr, sg, sb);
                }
            }
        }
        md /= cnt;
        double thr = Math.max(35.0, 3.0 * md);
        boolean[] raw = new boolean[n];
        for (int i = 0; i < n; i++) {
            raw[i] = dist(rgb[i], sr, sg, sb) > thr;
        }
        boolean[] mask = fillHoles(largestComponent(raw, w, h), w, h);
        int area = 0;
        for (boolean b : mask) {
            if (b) {
                area++;
            }
        }
        if (area < 0.01 * n) {
            boolean[] all = new boolean[n];
            java.util.Arrays.fill(all, true);
            return all;
        }
        return mask;
    }

    /**
     * Евклидово расстояние между цветом пикселя и заданным цветом в пространстве RGB.
     *
     * @param c пиксель
     * @param r красная составляющая эталона
     * @param g зелёная составляющая эталона
     * @param b синяя составляющая эталона
     * @return расстояние
     */
    private static double dist(int c, double r, double g, double b) {
        double dr = ((c >> 16) & 255) - r;
        double dg = ((c >> 8) & 255) - g;
        double db = (c & 255) - b;
        return Math.sqrt(dr * dr + dg * dg + db * db);
    }

    /**
     * Оставляет наибольшую 4-связную компоненту маски (обход в ширину).
     *
     * @param m маска
     * @param w ширина
     * @param h высота
     * @return маска наибольшей компоненты
     */
    private static boolean[] largestComponent(boolean[] m, int w, int h) {
        int n = w * h;
        int[] label = new int[n];
        int[] queue = new int[n];
        int cur = 0;
        int best = 0;
        int bestSize = 0;
        for (int s = 0; s < n; s++) {
            if (!m[s] || label[s] != 0) {
                continue;
            }
            cur++;
            int head = 0;
            int tail = 0;
            queue[tail++] = s;
            label[s] = cur;
            while (head < tail) {
                int p = queue[head++];
                int x = p % w;
                int y = p / w;
                if (x > 0 && m[p - 1] && label[p - 1] == 0) {
                    label[p - 1] = cur;
                    queue[tail++] = p - 1;
                }
                if (x < w - 1 && m[p + 1] && label[p + 1] == 0) {
                    label[p + 1] = cur;
                    queue[tail++] = p + 1;
                }
                if (y > 0 && m[p - w] && label[p - w] == 0) {
                    label[p - w] = cur;
                    queue[tail++] = p - w;
                }
                if (y < h - 1 && m[p + w] && label[p + w] == 0) {
                    label[p + w] = cur;
                    queue[tail++] = p + w;
                }
            }
            if (tail > bestSize) {
                bestSize = tail;
                best = cur;
            }
        }
        boolean[] out = new boolean[n];
        if (best != 0) {
            for (int i = 0; i < n; i++) {
                out[i] = label[i] == best;
            }
        }
        return out;
    }

    /**
     * Заполняет «дыры» маски: фон, недостижимый из границы кадра, присоединяется к объекту.
     *
     * @param m маска
     * @param w ширина
     * @param h высота
     * @return маска без внутренних дыр
     */
    private static boolean[] fillHoles(boolean[] m, int w, int h) {
        int n = w * h;
        boolean[] reach = new boolean[n];
        int[] queue = new int[n];
        int head = 0;
        int tail = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int p = y * w + x;
                boolean border = x == 0 || y == 0 || x == w - 1 || y == h - 1;
                if (border && !m[p] && !reach[p]) {
                    reach[p] = true;
                    queue[tail++] = p;
                }
            }
        }
        while (head < tail) {
            int p = queue[head++];
            int x = p % w;
            int y = p / w;
            if (x > 0 && !m[p - 1] && !reach[p - 1]) {
                reach[p - 1] = true;
                queue[tail++] = p - 1;
            }
            if (x < w - 1 && !m[p + 1] && !reach[p + 1]) {
                reach[p + 1] = true;
                queue[tail++] = p + 1;
            }
            if (y > 0 && !m[p - w] && !reach[p - w]) {
                reach[p - w] = true;
                queue[tail++] = p - w;
            }
            if (y < h - 1 && !m[p + w] && !reach[p + w]) {
                reach[p + w] = true;
                queue[tail++] = p + w;
            }
        }
        boolean[] out = new boolean[n];
        for (int i = 0; i < n; i++) {
            out[i] = m[i] || !reach[i];
        }
        return out;
    }

    /**
     * Вычисляет вектор признаков по пикселям и маске объекта.
     *
     * @param rgb  пиксели
     * @param w    ширина
     * @param h    высота
     * @param mask маска объекта (хотя бы один пиксель)
     * @return вектор признаков
     */
    private double[] compute(int[] rgb, int w, int h, boolean[] mask) {
        int n = w * h;
        int area = 0;
        int red = 0;
        int green = 0;
        int yellow = 0;
        double sinSum = 0;
        double cosSum = 0;
        double sSum = 0;
        double vSum = 0;
        double cx = 0;
        double cy = 0;
        int minX = w;
        int maxX = -1;
        int minY = h;
        int maxY = -1;
        double[] hist = new double[HIST_BINS];
        float[] gray = new float[n];
        for (int i = 0; i < n; i++) {
            int c = rgb[i];
            gray[i] = 0.299f * ((c >> 16) & 255) + 0.587f * ((c >> 8) & 255) + 0.114f * (c & 255);
            if (!mask[i]) {
                continue;
            }
            int x = i % w;
            int y = i / w;
            area++;
            cx += x;
            cy += y;
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
            float[] hsv = rgbToHsv(c);
            float hh = hsv[0];
            float s = hsv[1];
            float v = hsv[2];
            if ((hh <= 20f || hh >= 340f) && s > 0.4f && v > 0.2f) {
                red++;
            }
            if (hh >= 70f && hh <= 165f && s > 0.3f && v > 0.2f) {
                green++;
            }
            if (hh >= 40f && hh < 70f && s > 0.4f && v > 0.3f) {
                yellow++;
            }
            double rad = Math.toRadians(hh);
            sinSum += Math.sin(rad);
            cosSum += Math.cos(rad);
            sSum += s;
            vSum += v;
            hist[Math.min(HIST_BINS - 1, (int) (v * HIST_BINS))]++;
        }
        cx /= area;
        cy /= area;
        double maxD2 = 0;
        double gradSum = 0;
        int gradCnt = 0;
        for (int i = 0; i < n; i++) {
            if (!mask[i]) {
                continue;
            }
            int x = i % w;
            int y = i / w;
            boolean boundary = x == 0 || y == 0 || x == w - 1 || y == h - 1
                    || !mask[i - 1] || !mask[i + 1] || !mask[i - w] || !mask[i + w];
            if (boundary) {
                double dx = x - cx;
                double dy = y - cy;
                maxD2 = Math.max(maxD2, dx * dx + dy * dy);
            }
            if (x > 0 && y > 0 && x < w - 1 && y < h - 1) {
                gradSum += Math.abs(gray[i + 1] - gray[i - 1]) + Math.abs(gray[i + w] - gray[i - w]);
                gradCnt++;
            }
        }
        double rr = Math.sqrt(maxD2) + 0.5;
        double roundness = Math.min(1.0, area / (Math.PI * rr * rr));
        double bw = maxX - minX + 1;
        double bh = maxY - minY + 1;
        double aspect = bw / bh;
        double elong = Math.max(aspect, 1.0 / aspect);
        double hue = Math.toDegrees(Math.atan2(sinSum, cosSum));
        if (hue < 0) {
            hue += 360.0;
        }
        double rRatio = (double) red / area;
        double gRatio = (double) green / area;
        double yRatio = (double) yellow / area;
        double[] f = new double[NAMES.size()];
        f[IDX_RED] = rRatio;
        f[IDX_GREEN] = gRatio;
        f[IDX_YELLOW] = yRatio;
        f[IDX_DOMINANCE] = Math.max(rRatio, Math.max(gRatio, yRatio));
        f[IDX_HUE] = hue;
        f[IDX_SAT] = sSum / area;
        f[IDX_VAL] = vSum / area;
        f[IDX_ROUND] = roundness;
        f[IDX_ASPECT] = aspect;
        f[IDX_ELONG] = elong;
        f[IDX_TEXTURE] = gradCnt == 0 ? 0.0 : gradSum / (gradCnt * 2.0 * 255.0);
        for (int b = 0; b < HIST_BINS; b++) {
            f[IDX_HIST + b] = hist[b] / area;
        }
        return f;
    }
}
