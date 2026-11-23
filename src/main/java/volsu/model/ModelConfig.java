// src/main/java/volsu/model/ModelConfig.java
package volsu.model;

import volsu.FeatureExtractor;
import volsu.ModelException;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Модель распознавания: разбиение на классы, рабочий словарь признаков (3–5 признаков),
 * решающие границы и параметры уверенности. Сохраняется в JSON ({@code models/model_A*.json}).
 * <p>
 * Для каждого признака хранятся четыре числа {@code [strictLo, strictHi, looseLo, looseHi]}:
 * <ul>
 *   <li>«строгий» интервал — характерные значения яблока;</li>
 *   <li>«широкий» интервал — допустимые значения: выход за него означает «явно не яблоко».</li>
 * </ul>
 * Пусть s — число признаков в строгих интервалах, r — число признаков вне широких,
 * a = {@code minStrictVotes}, b = {@code rejectVotes}. Правило: r ≥ b → ΩA1_2 «явно не яблоко»;
 * иначе s ≥ a → ΩA1_1 «яблоко»; иначе → ΩA1_3 «похоже на яблоко».
 * По умолчанию a = числу признаков, b = 1 (классическое правило «все признаки в строгих интервалах»).
 */
public final class ModelConfig {

    /** Текущая версия формата модели. */
    public static final int CURRENT_VERSION = 1;

    private final int version;
    private final ClassPartition partition;
    private final List<String> featureNames;
    private final Map<String, double[]> thresholds;
    private final int resizeWidth;
    private final int resizeHeight;
    private final double unsureMinConfidence;
    private final double unsureMaxConfidence;
    private final int minStrictVotes;
    private final int rejectVotes;

    /**
     * Создаёт модель с параметрами по умолчанию: уверенность 0.3 … 0.5 для класса ΩA1_3,
     * правило «все признаки в строгих интервалах».
     *
     * @param partition    разбиение
     * @param featureNames имена признаков рабочего словаря
     * @param thresholds   границы {@code [strictLo, strictHi, looseLo, looseHi]} по каждому признаку
     * @param resizeWidth  ширина масштабирования
     * @param resizeHeight высота масштабирования
     */
    public ModelConfig(ClassPartition partition, List<String> featureNames,
                       Map<String, double[]> thresholds, int resizeWidth, int resizeHeight) {
        this(CURRENT_VERSION, partition, featureNames, thresholds, resizeWidth, resizeHeight, 0.3, 0.5);
    }

    /**
     * Конструктор с параметрами уверенности и правилом по умолчанию
     * (a = числу признаков, b = 1).
     *
     * @param version             версия формата
     * @param partition           разбиение
     * @param featureNames        имена признаков
     * @param thresholds          границы по признакам
     * @param resizeWidth         ширина масштабирования
     * @param resizeHeight        высота масштабирования
     * @param unsureMinConfidence минимальная уверенность для класса «похоже на яблоко»
     * @param unsureMaxConfidence максимальная уверенность для класса «похоже на яблоко»
     */
    public ModelConfig(int version, ClassPartition partition, List<String> featureNames,
                       Map<String, double[]> thresholds, int resizeWidth, int resizeHeight,
                       double unsureMinConfidence, double unsureMaxConfidence) {
        this(version, partition, featureNames, thresholds, resizeWidth, resizeHeight,
                unsureMinConfidence, unsureMaxConfidence, featureNames.size(), 1);
    }

    /**
     * Полный конструктор.
     *
     * @param version             версия формата
     * @param partition           разбиение
     * @param featureNames        имена признаков
     * @param thresholds          границы по признакам
     * @param resizeWidth         ширина масштабирования
     * @param resizeHeight        высота масштабирования
     * @param unsureMinConfidence минимальная уверенность для класса «похоже на яблоко»
     * @param unsureMaxConfidence максимальная уверенность для класса «похоже на яблоко»
     * @param minStrictVotes      минимальное число признаков в строгих интервалах для класса «яблоко»
     * @param rejectVotes         число признаков вне широких интервалов, при котором объект «явно не яблоко»
     */
    public ModelConfig(int version, ClassPartition partition, List<String> featureNames,
                       Map<String, double[]> thresholds, int resizeWidth, int resizeHeight,
                       double unsureMinConfidence, double unsureMaxConfidence,
                       int minStrictVotes, int rejectVotes) {
        this.version = version;
        this.partition = partition;
        this.featureNames = Collections.unmodifiableList(new ArrayList<>(featureNames));
        Map<String, double[]> copy = new LinkedHashMap<>();
        for (Map.Entry<String, double[]> e : thresholds.entrySet()) {
            copy.put(e.getKey(), e.getValue().clone());
        }
        this.thresholds = Collections.unmodifiableMap(copy);
        this.resizeWidth = resizeWidth;
        this.resizeHeight = resizeHeight;
        this.unsureMinConfidence = unsureMinConfidence;
        this.unsureMaxConfidence = unsureMaxConfidence;
        this.minStrictVotes = minStrictVotes;
        this.rejectVotes = rejectVotes;
    }

    /**
     * Возвращает копию модели с другими параметрами уверенности класса ΩA1_3.
     *
     * @param min новая минимальная уверенность
     * @param max новая максимальная уверенность
     * @return новая модель
     */
    public ModelConfig withUnsureConfidence(double min, double max) {
        return new ModelConfig(version, partition, featureNames, thresholds,
                resizeWidth, resizeHeight, min, max, minStrictVotes, rejectVotes);
    }

    /**
     * Версия формата.
     *
     * @return версия
     */
    public int getVersion() {
        return version;
    }

    /**
     * Разбиение модели.
     *
     * @return разбиение
     */
    public ClassPartition getPartition() {
        return partition;
    }

    /**
     * Рабочий словарь признаков.
     *
     * @return неизменяемый список имён признаков
     */
    public List<String> getFeatures() {
        return featureNames;
    }

    /**
     * Решающие границы.
     *
     * @return карта «признак → [strictLo, strictHi, looseLo, looseHi]»
     */
    public Map<String, double[]> getThresholds() {
        return thresholds;
    }

    /**
     * Ширина масштабирования.
     *
     * @return пиксели
     */
    public int getResizeWidth() {
        return resizeWidth;
    }

    /**
     * Высота масштабирования.
     *
     * @return пиксели
     */
    public int getResizeHeight() {
        return resizeHeight;
    }

    /**
     * Минимальная уверенность для класса ΩA1_3.
     *
     * @return значение 0..1
     */
    public double getUnsureMinConfidence() {
        return unsureMinConfidence;
    }

    /**
     * Максимальная уверенность для класса ΩA1_3.
     *
     * @return значение 0..1
     */
    public double getUnsureMaxConfidence() {
        return unsureMaxConfidence;
    }

    /**
     * Минимальное число признаков в строгих интервалах для класса «яблоко».
     *
     * @return значение от 1 до числа признаков
     */
    public int getMinStrictVotes() {
        return minStrictVotes;
    }

    /**
     * Число признаков вне широких интервалов, при котором объект относится к «явно не яблоку».
     *
     * @return значение от 1 до числа признаков
     */
    public int getRejectVotes() {
        return rejectVotes;
    }

    /**
     * Проверяет целостность модели: версию, непустоту словаря, известность признаков,
     * наличие и упорядоченность границ, корректность параметров.
     *
     * @throws ModelException если модель некорректна (→ код возврата 2)
     */
    public void validate() throws ModelException {
        if (version != CURRENT_VERSION) {
            throw new ModelException("Неподдерживаемая версия модели: " + version);
        }
        if (partition == null) {
            throw new ModelException("В модели не задано разбиение");
        }
        if (featureNames.isEmpty()) {
            throw new ModelException("Рабочий словарь признаков пуст");
        }
        if (resizeWidth <= 0 || resizeHeight <= 0) {
            throw new ModelException("Некорректный размер масштабирования");
        }
        if (!(unsureMinConfidence >= 0 && unsureMaxConfidence <= 1 && unsureMinConfidence <= unsureMaxConfidence)) {
            throw new ModelException("Некорректные параметры уверенности");
        }
        int n = featureNames.size();
        if (minStrictVotes < 1 || minStrictVotes > n) {
            throw new ModelException("Некорректное число признаков для класса «яблоко»: " + minStrictVotes);
        }
        if (rejectVotes < 1 || rejectVotes > n) {
            throw new ModelException("Некорректное число признаков для отклонения: " + rejectVotes);
        }
        for (String f : featureNames) {
            if (!FeatureExtractor.featureNames().contains(f)) {
                throw new ModelException("Неизвестный признак в модели: " + f);
            }
            double[] t = thresholds.get(f);
            if (t == null || t.length != 4) {
                throw new ModelException("Для признака " + f + " не заданы границы (нужно 4 числа)");
            }
            for (double v : t) {
                if (Double.isNaN(v) || Double.isInfinite(v)) {
                    throw new ModelException("Недопустимое значение границы признака " + f);
                }
            }
            double eps = 1e-9;
            if (t[0] > t[1] || t[2] > t[3] || t[2] > t[0] + eps || t[1] > t[3] + eps) {
                throw new ModelException("Нарушена вложенность интервалов признака " + f);
            }
        }
    }

    /**
     * Сериализует модель в JSON.
     *
     * @return текст JSON
     */
    public String toJson() {
        StringBuilder sb = new StringBuilder("{\n");
        sb.append("  \"version\": ").append(version).append(",\n");
        sb.append("  \"partitionName\": ").append(quote(partition.name())).append(",\n");
        sb.append("  \"classNames\": ").append(stringArray(partition.classes())).append(",\n");
        sb.append("  \"decisions\": ").append(intArray(partition.getDecisions())).append(",\n");
        sb.append("  \"baseToLocal\": ").append(intArray(partition.getBaseToLocal())).append(",\n");
        sb.append("  \"resizeWidth\": ").append(resizeWidth).append(",\n");
        sb.append("  \"resizeHeight\": ").append(resizeHeight).append(",\n");
        sb.append("  \"unsureMinConfidence\": ").append(unsureMinConfidence).append(",\n");
        sb.append("  \"unsureMaxConfidence\": ").append(unsureMaxConfidence).append(",\n");
        sb.append("  \"minStrictVotes\": ").append(minStrictVotes).append(",\n");
        sb.append("  \"rejectVotes\": ").append(rejectVotes).append(",\n");
        sb.append("  \"featureNames\": ").append(stringArray(featureNames)).append(",\n");
        sb.append("  \"thresholds\": {\n");
        int i = 0;
        for (Map.Entry<String, double[]> e : thresholds.entrySet()) {
            double[] t = e.getValue();
            sb.append("    ").append(quote(e.getKey())).append(": [")
                    .append(t[0]).append(", ").append(t[1]).append(", ")
                    .append(t[2]).append(", ").append(t[3]).append("]");
            sb.append(++i < thresholds.size() ? ",\n" : "\n");
        }
        sb.append("  }\n}\n");
        return sb.toString();
    }

    /**
     * Строит модель из JSON и проверяет её целостность. Поля {@code minStrictVotes} и
     * {@code rejectVotes} необязательны (для моделей старого формата берутся значения по умолчанию).
     *
     * @param json текст JSON
     * @return модель
     * @throws ModelException при ошибках разбора или валидации
     */
    public static ModelConfig fromJson(String json) throws ModelException {
        Object root;
        try {
            root = new JsonParser(json).parse();
        } catch (IllegalArgumentException e) {
            throw new ModelException("Некорректный JSON модели: " + e.getMessage(), e);
        }
        if (!(root instanceof Map)) {
            throw new ModelException("Корень JSON модели должен быть объектом");
        }
        Map<?, ?> m = (Map<?, ?>) root;
        ModelConfig cfg;
        try {
            ClassPartition partition = new ClassPartition(
                    (String) req(m, "partitionName"),
                    strList(req(m, "classNames")),
                    intArr(req(m, "decisions")),
                    intArr(req(m, "baseToLocal")));
            List<String> features = strList(req(m, "featureNames"));
            Map<String, double[]> th = new LinkedHashMap<>();
            Map<?, ?> thMap = (Map<?, ?>) req(m, "thresholds");
            for (Map.Entry<?, ?> e : thMap.entrySet()) {
                th.put((String) e.getKey(), dblArr(e.getValue()));
            }
            int strictVotes = m.get("minStrictVotes") != null
                    ? ((Number) m.get("minStrictVotes")).intValue() : features.size();
            int rejectVotes = m.get("rejectVotes") != null
                    ? ((Number) m.get("rejectVotes")).intValue() : 1;
            cfg = new ModelConfig(((Number) req(m, "version")).intValue(), partition, features, th,
                    ((Number) req(m, "resizeWidth")).intValue(),
                    ((Number) req(m, "resizeHeight")).intValue(),
                    ((Number) req(m, "unsureMinConfidence")).doubleValue(),
                    ((Number) req(m, "unsureMaxConfidence")).doubleValue(),
                    strictVotes, rejectVotes);
        } catch (ClassCastException | IllegalArgumentException e) {
            throw new ModelException("Структура модели повреждена: " + e.getMessage(), e);
        }
        cfg.validate();
        return cfg;
    }

    /**
     * Записывает модель в файл (UTF-8), создавая каталоги.
     *
     * @param path путь к файлу, например {@code models/model_A1.json}
     * @throws IOException при ошибке записи
     */
    public void save(Path path) throws IOException {
        try {
            validate();
        } catch (ModelException e) {
            throw new IOException("Попытка сохранить некорректную модель: " + e.getMessage(), e);
        }
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Files.writeString(path, toJson(), StandardCharsets.UTF_8);
    }

    /**
     * Читает модель из файла.
     *
     * @param path путь к JSON-файлу
     * @return модель
     * @throws ModelException если файл не читается или модель некорректна (→ код 2)
     */
    public static ModelConfig load(Path path) throws ModelException {
        String text;
        try {
            text = Files.readString(path, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new ModelException("Не удалось прочитать модель: " + path, e);
        }
        return fromJson(text);
    }

    /* ----------------------------- вспомогательные методы ----------------------------- */

    /**
     * Возвращает обязательное поле JSON-объекта.
     *
     * @param m   объект
     * @param key имя поля
     * @return значение поля
     */
    private static Object req(Map<?, ?> m, String key) {
        Object v = m.get(key);
        if (v == null) {
            throw new IllegalArgumentException("отсутствует поле \"" + key + "\"");
        }
        return v;
    }

    /**
     * Преобразует JSON-массив в список строк.
     *
     * @param o значение JSON
     * @return список строк
     */
    private static List<String> strList(Object o) {
        List<String> res = new ArrayList<>();
        for (Object x : (List<?>) o) {
            res.add((String) x);
        }
        return res;
    }

    /**
     * Преобразует JSON-массив в массив int.
     *
     * @param o значение JSON
     * @return массив целых
     */
    private static int[] intArr(Object o) {
        List<?> l = (List<?>) o;
        int[] r = new int[l.size()];
        for (int i = 0; i < r.length; i++) {
            r[i] = ((Number) l.get(i)).intValue();
        }
        return r;
    }

    /**
     * Преобразует JSON-массив в массив double.
     *
     * @param o значение JSON
     * @return массив вещественных
     */
    private static double[] dblArr(Object o) {
        List<?> l = (List<?>) o;
        double[] r = new double[l.size()];
        for (int i = 0; i < r.length; i++) {
            r[i] = ((Number) l.get(i)).doubleValue();
        }
        return r;
    }

    /**
     * Экранирует строку для JSON.
     *
     * @param s исходная строка
     * @return строка в кавычках
     */
    private static String quote(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"':
                    sb.append("\\\"");
                    break;
                case '\\':
                    sb.append("\\\\");
                    break;
                case '\n':
                    sb.append("\\n");
                    break;
                case '\t':
                    sb.append("\\t");
                    break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        return sb.append('"').toString();
    }

    /**
     * Формирует JSON-массив строк.
     *
     * @param list список строк
     * @return текст массива
     */
    private static String stringArray(List<String> list) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < list.size(); i++) {
            sb.append(i > 0 ? ", " : "").append(quote(list.get(i)));
        }
        return sb.append(']').toString();
    }

    /**
     * Формирует JSON-массив целых.
     *
     * @param a массив
     * @return текст массива
     */
    private static String intArray(int[] a) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < a.length; i++) {
            sb.append(i > 0 ? ", " : "").append(a[i]);
        }
        return sb.append(']').toString();
    }

    /**
     * Минимальный рекурсивный JSON-парсер (объекты, массивы, строки, числа, true/false/null).
     */
    private static final class JsonParser {
        private final String s;
        private int i;

        /**
         * Создаёт парсер.
         *
         * @param s текст JSON
         */
        JsonParser(String s) {
            this.s = s == null ? "" : s;
        }

        /**
         * Разбирает весь текст как одно значение.
         *
         * @return Map, List, String, Double, Boolean или null
         */
        Object parse() {
            Object v = value();
            skipWs();
            if (i != s.length()) {
                throw error("лишние символы после значения");
            }
            return v;
        }

        /**
         * Формирует исключение с позицией.
         *
         * @param msg сообщение
         * @return исключение
         */
        private IllegalArgumentException error(String msg) {
            return new IllegalArgumentException(msg + " (позиция " + i + ")");
        }

        /** Пропускает пробельные символы. */
        private void skipWs() {
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) {
                i++;
            }
        }

        /**
         * Требует конкретный символ.
         *
         * @param c ожидаемый символ
         */
        private void expect(char c) {
            skipWs();
            if (i >= s.length() || s.charAt(i) != c) {
                throw error("ожидался символ '" + c + "'");
            }
            i++;
        }

        /**
         * Разбирает значение произвольного типа.
         *
         * @return значение
         */
        private Object value() {
            skipWs();
            if (i >= s.length()) {
                throw error("неожиданный конец текста");
            }
            char c = s.charAt(i);
            if (c == '{') {
                return object();
            }
            if (c == '[') {
                return array();
            }
            if (c == '"') {
                return string();
            }
            if (c == '-' || Character.isDigit(c)) {
                return number();
            }
            if (s.startsWith("true", i)) {
                i += 4;
                return Boolean.TRUE;
            }
            if (s.startsWith("false", i)) {
                i += 5;
                return Boolean.FALSE;
            }
            if (s.startsWith("null", i)) {
                i += 4;
                return null;
            }
            throw error("неожиданный символ '" + c + "'");
        }

        /**
         * Разбирает объект.
         *
         * @return карта «ключ → значение»
         */
        private Map<String, Object> object() {
            Map<String, Object> map = new LinkedHashMap<>();
            expect('{');
            skipWs();
            if (i < s.length() && s.charAt(i) == '}') {
                i++;
                return map;
            }
            while (true) {
                skipWs();
                String key = string();
                expect(':');
                map.put(key, value());
                skipWs();
                if (i < s.length() && s.charAt(i) == ',') {
                    i++;
                    continue;
                }
                expect('}');
                return map;
            }
        }

        /**
         * Разбирает массив.
         *
         * @return список значений
         */
        private List<Object> array() {
            List<Object> list = new ArrayList<>();
            expect('[');
            skipWs();
            if (i < s.length() && s.charAt(i) == ']') {
                i++;
                return list;
            }
            while (true) {
                list.add(value());
                skipWs();
                if (i < s.length() && s.charAt(i) == ',') {
                    i++;
                    continue;
                }
                expect(']');
                return list;
            }
        }

        /**
         * Разбирает строку со стандартными escape-последовательностями.
         *
         * @return строка
         */
        private String string() {
            expect('"');
            StringBuilder sb = new StringBuilder();
            while (true) {
                if (i >= s.length()) {
                    throw error("незакрытая строка");
                }
                char c = s.charAt(i++);
                if (c == '"') {
                    return sb.toString();
                }
                if (c == '\\') {
                    if (i >= s.length()) {
                        throw error("обрыв escape-последовательности");
                    }
                    char e = s.charAt(i++);
                    switch (e) {
                        case 'n':
                            sb.append('\n');
                            break;
                        case 't':
                            sb.append('\t');
                            break;
                        case 'r':
                            sb.append('\r');
                            break;
                        case 'b':
                            sb.append('\b');
                            break;
                        case 'f':
                            sb.append('\f');
                            break;
                        case 'u':
                            if (i + 4 > s.length()) {
                                throw error("неполная \\u-последовательность");
                            }
                            sb.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                            i += 4;
                            break;
                        default:
                            sb.append(e);
                    }
                } else {
                    sb.append(c);
                }
            }
        }

        /**
         * Разбирает число.
         *
         * @return значение типа Double
         */
        private Double number() {
            int start = i;
            while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) {
                i++;
            }
            try {
                return Double.parseDouble(s.substring(start, i));
            } catch (NumberFormatException ex) {
                throw error("некорректное число");
            }
        }
    }
}
