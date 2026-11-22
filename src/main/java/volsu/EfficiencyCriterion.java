// src/main/java/volsu/EfficiencyCriterion.java
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
 * Критерий эффективности системы распознавания
 * <pre>R = M[G(Ω)] = Σ<sub>i</sub> P(Ωi | x) · G(Ωi).</pre>
 * Реализация по матрице ошибок. Выигрыш за пару «истинный объект — принятое решение»:
 * правильно распознанное яблоко +gainApple, правильно распознанное «не яблоко» +gainNotApple,
 * ложное срабатывание — штраф penaltyFalseAlarm (&lt; 0), пропуск яблока — штраф penaltyMiss (&lt; 0).
 * <p>
 * Вклад класса Ωi: {@code G(Ωi) = (1/N) · Σ<sub>t</sub> n(t, i) · payoff(t, d(i))}, где n(t, i) —
 * число объектов истинного типа t, отнесённых к классу i, d(i) — решение, сопоставленное классу.
 * Тогда {@code R = Σ G(Ωi)} — средний выигрыш на один объект.
 * <p>
 * Шаг уточнения (этап 2): худший класс v = argmin G(Ωi); его объекты переносятся в класс μ,
 * для которого потеря {@code ΔG(μ) = G(Ωv) + G(Ωμ) − G'(Ωμ)} минимальна
 * (G' — вклад класса μ после присоединения объектов v).
 */
public class EfficiencyCriterion {

    private double gainApple = 10.0;
    private double gainNotApple = 5.0;
    private double penaltyFalseAlarm = -8.0;
    private double penaltyMiss = -10.0;
    private final List<Analysis> results = new ArrayList<>();

    /**
     * Итог анализа одного разбиения.
     */
    public static final class Analysis {
        /** Имя разбиения. */
        public String name;
        /** Матрица ошибок. */
        public ConfusionMatrix cm;
        /** G(Ωi) по классам. */
        public double[] gains;
        /** Суммарные штрафы по классам (в расчёте на объект). */
        public double[] penalties;
        /** Минимальное ΔG при переносе объектов класса i в другой класс (NaN, если класс один). */
        public double[] deltaMin;
        /** Класс-получатель с минимальным ΔG для каждого класса (−1, если нет). */
        public int[] mergeTo;
        /** Значение критерия R. */
        public double r;
        /** Индекс худшего класса (минимум G). */
        public int worstClass;
        /** Индекс класса μ для переноса объектов худшего класса (−1, если нет). */
        public int mergeTarget;
    }

    /** Создаёт критерий с ценами по умолчанию: +10, +5, штрафы −8 и −10. */
    public EfficiencyCriterion() {
    }

    /**
     * Создаёт критерий с заданными ценами.
     *
     * @param gainApple         выигрыш за правильное распознавание яблока
     * @param gainNotApple      выигрыш за правильное распознавание «не яблока»
     * @param penaltyFalseAlarm штраф за ложное срабатывание
     * @param penaltyMiss       штраф за пропуск яблока
     */
    public EfficiencyCriterion(double gainApple, double gainNotApple,
                               double penaltyFalseAlarm, double penaltyMiss) {
        defineGains(gainApple, gainNotApple);
        definePenalties(penaltyFalseAlarm, penaltyMiss);
    }

    /**
     * Задаёт цены правильного распознавания (берутся по модулю).
     *
     * @param apple    выигрыш за правильное распознавание яблока
     * @param notApple выигрыш за правильное распознавание «не яблока»
     */
    public void defineGains(double apple, double notApple) {
        this.gainApple = Math.abs(apple);
        this.gainNotApple = Math.abs(notApple);
    }

    /**
     * Задаёт штрафы за ошибки (величины берутся по модулю и применяются со знаком «минус»).
     *
     * @param falseAlarm штраф за ложное срабатывание (не-яблоко принято за яблоко)
     * @param miss       штраф за пропуск (яблоко принято за не-яблоко)
     */
    public void definePenalties(double falseAlarm, double miss) {
        this.penaltyFalseAlarm = -Math.abs(falseAlarm);
        this.penaltyMiss = -Math.abs(miss);
    }

    /**
     * Выигрыш (штраф) за пару «истинная метка — решение».
     *
     * @param trueLabel истинная метка (0 — яблоко, 1 — не яблоко)
     * @param decision  принятое решение (0 — l1, 1 — l2)
     * @return выигрыш или отрицательный штраф
     */
    private double payoff(int trueLabel, int decision) {
        if (trueLabel == Sample.APPLE) {
            return decision == ClassPartition.DECISION_APPLE ? gainApple : penaltyMiss;
        }
        return decision == ClassPartition.DECISION_NOT_APPLE ? gainNotApple : penaltyFalseAlarm;
    }

    /**
     * Вклад каждого класса G(Ωi) в критерий (сумма выигрышей и штрафов, делённая на N).
     *
     * @param cm матрица ошибок
     * @return массив G по классам
     */
    public double[] gainByClass(ConfusionMatrix cm) {
        int m = cm.classCount();
        double[] g = new double[m];
        int n = cm.total();
        if (n == 0) {
            return g;
        }
        for (int i = 0; i < m; i++) {
            int d = cm.partition().decisionFor(i);
            double sum = 0;
            for (int t = 0; t < 2; t++) {
                sum += cm.get(t, i) * payoff(t, d);
            }
            g[i] = sum / n;
        }
        return g;
    }

    /**
     * Суммарные штрафы (только отрицательные выплаты) по классам в расчёте на один объект.
     *
     * @param cm матрица ошибок
     * @return массив штрафов (≤ 0)
     */
    public double[] penaltyByClass(ConfusionMatrix cm) {
        int m = cm.classCount();
        double[] p = new double[m];
        int n = cm.total();
        if (n == 0) {
            return p;
        }
        for (int i = 0; i < m; i++) {
            int d = cm.partition().decisionFor(i);
            double sum = 0;
            for (int t = 0; t < 2; t++) {
                double pay = payoff(t, d);
                if (pay < 0) {
                    sum += cm.get(t, i) * pay;
                }
            }
            p[i] = sum / n;
        }
        return p;
    }

    /**
     * Критерий эффективности R = Σ G(Ωi) для текущего разбиения.
     *
     * @param cm матрица ошибок
     * @return R (средний выигрыш на объект)
     */
    public double computeR(ConfusionMatrix cm) {
        double r = 0;
        for (double g : gainByClass(cm)) {
            r += g;
        }
        return r;
    }

    /**
     * Находит класс ΩA1v с минимальным G.
     *
     * @param gains значения G по классам
     * @return индекс худшего класса
     */
    public int findWorstClass(double[] gains) {
        int worst = 0;
        for (int i = 1; i < gains.length; i++) {
            if (gains[i] < gains[worst]) {
                worst = i;
            }
        }
        return worst;
    }

    /**
     * Потеря ΔG при переносе объектов класса {@code v} в класс {@code mu}:
     * {@code ΔG = G(Ωv) + G(Ωμ) − G'(Ωμ)}.
     *
     * @param cm матрица ошибок
     * @param v  исключаемый класс
     * @param mu класс-получатель
     * @return потеря выигрыша (меньше — лучше)
     */
    public double deltaG(ConfusionMatrix cm, int v, int mu) {
        int n = cm.total();
        if (n == 0) {
            return 0.0;
        }
        double[] g = gainByClass(cm);
        int d = cm.partition().decisionFor(mu);
        double merged = 0;
        for (int t = 0; t < 2; t++) {
            merged += (cm.get(t, v) + cm.get(t, mu)) * payoff(t, d);
        }
        merged /= n;
        return g[v] + g[mu] - merged;
    }

    /**
     * Выбирает класс μ ≠ v с минимальным ΔG.
     *
     * @param cm матрица ошибок
     * @param v  исключаемый класс
     * @return пара {v, μ} для построения следующего разбиения
     * @throws IllegalArgumentException если в разбиении один класс
     */
    public int[] findBestMerge(ConfusionMatrix cm, int v) {
        int m = cm.classCount();
        if (m < 2) {
            throw new IllegalArgumentException("В разбиении один класс — объединять нечего");
        }
        int best = -1;
        double bestDelta = Double.POSITIVE_INFINITY;
        for (int mu = 0; mu < m; mu++) {
            if (mu == v) {
                continue;
            }
            double d = deltaG(cm, v, mu);
            if (d < bestDelta) {
                bestDelta = d;
                best = mu;
            }
        }
        return new int[]{v, best};
    }

    /**
     * Выбирает лучшее разбиение по максимуму R (при равенстве — более позднее, то есть более простое).
     *
     * @param rValues значения R⁽¹⁾, R⁽²⁾, ...
     * @return индекс лучшего разбиения
     */
    public int chooseBestPartition(List<Double> rValues) {
        int best = 0;
        for (int i = 1; i < rValues.size(); i++) {
            if (rValues.get(i) >= rValues.get(best) - 1e-12) {
                best = i;
            }
        }
        return best;
    }

    /**
     * Полный анализ разбиения: G, штрафы, ΔG, R, худший класс и класс-получатель.
     * Результат добавляется во внутренний список (для {@link #saveGains(String)} и
     * {@link #saveSummary(String)}).
     *
     * @param name имя разбиения
     * @param cm   матрица ошибок
     * @return результат анализа
     */
    public Analysis analyze(String name, ConfusionMatrix cm) {
        int m = cm.classCount();
        Analysis a = new Analysis();
        a.name = name;
        a.cm = cm;
        a.gains = gainByClass(cm);
        a.penalties = penaltyByClass(cm);
        a.r = 0;
        for (double g : a.gains) {
            a.r += g;
        }
        a.worstClass = findWorstClass(a.gains);
        a.deltaMin = new double[m];
        a.mergeTo = new int[m];
        for (int i = 0; i < m; i++) {
            if (m > 1) {
                int[] best = findBestMerge(cm, i);
                a.mergeTo[i] = best[1];
                a.deltaMin[i] = deltaG(cm, i, best[1]);
            } else {
                a.mergeTo[i] = -1;
                a.deltaMin[i] = Double.NaN;
            }
        }
        a.mergeTarget = a.mergeTo[a.worstClass];
        results.add(a);
        return a;
    }

    /**
     * Накопленные результаты анализа.
     *
     * @return список результатов
     */
    public List<Analysis> getResults() {
        return results;
    }

    /**
     * Индекс лучшего из проанализированных разбиений по максимуму R.
     *
     * @return индекс в {@link #getResults()}
     */
    public int bestIndex() {
        List<Double> r = new ArrayList<>();
        for (Analysis a : results) {
            r.add(a.r);
        }
        return chooseBestPartition(r);
    }

    /** Очищает накопленные результаты. */
    public void reset() {
        results.clear();
    }

    /**
     * Записывает {@code results/gains_G.csv}: разбиение, класс, G, штрафы, ΔG, класс-получатель.
     *
     * @param path путь к файлу
     * @throws IOException при ошибке записи
     */
    public void saveGains(String path) throws IOException {
        StringBuilder sb = new StringBuilder("partition,class,G,penalty,deltaG,merge_to\n");
        for (Analysis a : results) {
            List<String> names = a.cm.partition().classes();
            for (int i = 0; i < a.gains.length; i++) {
                String delta = Double.isNaN(a.deltaMin[i]) ? "NA"
                        : String.format(Locale.ROOT, "%.6f", a.deltaMin[i]);
                String to = a.mergeTo[i] < 0 ? "NA" : names.get(a.mergeTo[i]);
                sb.append(String.format(Locale.ROOT, "%s,%s,%.6f,%.6f,%s,%s%n",
                        a.name, names.get(i), a.gains[i], a.penalties[i], delta, to));
            }
        }
        write(path, sb.toString());
    }

    /**
     * Записывает {@code results/R_summary.csv}: разбиение, R, худший класс, класс-получатель,
     * ΔG и пометку лучшего варианта.
     *
     * @param path путь к файлу
     * @throws IOException при ошибке записи
     */
    public void saveSummary(String path) throws IOException {
        StringBuilder sb = new StringBuilder("partition,R,worst_class,merge_to,deltaG,best\n");
        int best = results.isEmpty() ? -1 : bestIndex();
        for (int k = 0; k < results.size(); k++) {
            Analysis a = results.get(k);
            List<String> names = a.cm.partition().classes();
            String to = a.mergeTarget < 0 ? "NA" : names.get(a.mergeTarget);
            double d = a.deltaMin[a.worstClass];
            String delta = Double.isNaN(d) ? "NA" : String.format(Locale.ROOT, "%.6f", d);
            sb.append(String.format(Locale.ROOT, "%s,%.6f,%s,%s,%s,%s%n", a.name, a.r,
                    names.get(a.worstClass), to, delta, k == best ? "yes" : "no"));
        }
        write(path, sb.toString());
    }

    /**
     * Записывает текст в файл (UTF-8), создавая каталоги.
     *
     * @param path путь
     * @param text содержимое
     * @throws IOException при ошибке записи
     */
    private static void write(String path, String text) throws IOException {
        Path p = Path.of(path);
        Path parent = p.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Files.writeString(p, text, StandardCharsets.UTF_8);
    }
}
