// src/main/java/volsu/model/ClassPartition.java
package volsu.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Разбиение множества объектов на классы (A1, A2, ...) и соответствие классов решениям
 * из множества L = {l1 = «яблоко», l2 = «не яблоко»}.
 * <p>
 * Первое приближение A1 содержит m = k + 1 = 3 класса:
 * ΩA1_1 «яблоко», ΩA1_2 «явно не яблоко», ΩA1_3 «похоже на яблоко».
 * Последующие разбиения получаются исключением худшего класса ΩA1v
 * и переносом его объектов в класс ΩA1μ ({@link #merge(int, int)}).
 * <p>
 * Для работы классификатора хранится отображение {@code baseToLocal}: индекс класса
 * базового разбиения A1 → индекс класса данного разбиения.
 */
public final class ClassPartition {

    /** Индекс решения l1 («яблоко»). */
    public static final int DECISION_APPLE = 0;
    /** Индекс решения l2 («не яблоко»). */
    public static final int DECISION_NOT_APPLE = 1;

    /** Первое приближение: три класса (m = k + 1). */
    public static final ClassPartition A1 = new ClassPartition("A1",
            List.of("ΩA1_1 яблоко", "ΩA1_2 явно не яблоко", "ΩA1_3 похоже на яблоко"),
            new int[]{DECISION_APPLE, DECISION_NOT_APPLE, DECISION_NOT_APPLE},
            new int[]{0, 1, 2});

    private final String name;
    private final List<String> classNames;
    private final int[] decisions;
    private final int[] baseToLocal;

    /**
     * Создаёт разбиение.
     *
     * @param name        имя разбиения (A1, A2, ...)
     * @param classNames  имена классов
     * @param decisions   решение для каждого класса (0 — l1, 1 — l2)
     * @param baseToLocal отображение классов базового разбиения A1 в классы данного
     * @throws IllegalArgumentException при несогласованных аргументах
     */
    public ClassPartition(String name, List<String> classNames, int[] decisions, int[] baseToLocal) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("Не задано имя разбиения");
        }
        if (classNames == null || classNames.isEmpty()) {
            throw new IllegalArgumentException("Список классов пуст");
        }
        if (decisions == null || decisions.length != classNames.size()) {
            throw new IllegalArgumentException("Число решений не совпадает с числом классов");
        }
        for (int d : decisions) {
            if (d != DECISION_APPLE && d != DECISION_NOT_APPLE) {
                throw new IllegalArgumentException("Неизвестное решение: " + d);
            }
        }
        if (baseToLocal == null) {
            throw new IllegalArgumentException("Не задано отображение классов");
        }
        for (int b : baseToLocal) {
            if (b < 0 || b >= classNames.size()) {
                throw new IllegalArgumentException("Отображение ссылается на несуществующий класс: " + b);
            }
        }
        this.name = name;
        this.classNames = Collections.unmodifiableList(new ArrayList<>(classNames));
        this.decisions = decisions.clone();
        this.baseToLocal = baseToLocal.clone();
    }

    /**
     * Возвращает имя разбиения.
     *
     * @return имя (A1, A2, ...)
     */
    public String name() {
        return name;
    }

    /**
     * Возвращает список имён классов разбиения.
     *
     * @return неизменяемый список имён
     */
    public List<String> classes() {
        return classNames;
    }

    /**
     * Возвращает число классов разбиения.
     *
     * @return m
     */
    public int size() {
        return classNames.size();
    }

    /**
     * Сопоставляет классу решение из L.
     *
     * @param classIndex индекс класса
     * @return {@link #DECISION_APPLE} (l1) или {@link #DECISION_NOT_APPLE} (l2)
     */
    public int decisionFor(int classIndex) {
        return decisions[classIndex];
    }

    /**
     * Возвращает копию массива решений по классам.
     *
     * @return массив решений
     */
    public int[] getDecisions() {
        return decisions.clone();
    }

    /**
     * Возвращает копию отображения «класс A1 → класс данного разбиения».
     *
     * @return массив отображения
     */
    public int[] getBaseToLocal() {
        return baseToLocal.clone();
    }

    /**
     * Переводит индекс класса базового разбиения A1 в индекс класса данного разбиения.
     *
     * @param baseClass индекс класса в A1
     * @return индекс класса в данном разбиении
     */
    public int baseToLocal(int baseClass) {
        return baseToLocal[baseClass];
    }

    /**
     * Строит разбиение A(n+1) из A(n): класс {@code fromClass} исключается из алфавита,
     * его объекты относятся к классу {@code toClass}.
     *
     * @param fromClass индекс исключаемого класса ΩAnv
     * @param toClass   индекс класса ΩAnμ, принимающего объекты
     * @return новое разбиение
     * @throws IllegalArgumentException при неверных индексах или совпадении индексов
     */
    public ClassPartition merge(int fromClass, int toClass) {
        int m = size();
        if (m < 2) {
            throw new IllegalArgumentException("Нечего объединять: в разбиении один класс");
        }
        if (fromClass < 0 || fromClass >= m || toClass < 0 || toClass >= m || fromClass == toClass) {
            throw new IllegalArgumentException("Некорректные индексы объединения: " + fromClass + " -> " + toClass);
        }
        List<String> newNames = new ArrayList<>(classNames);
        newNames.remove(fromClass);
        int[] newDecisions = new int[m - 1];
        for (int i = 0, j = 0; i < m; i++) {
            if (i != fromClass) {
                newDecisions[j++] = decisions[i];
            }
        }
        int[] newMap = new int[baseToLocal.length];
        for (int b = 0; b < baseToLocal.length; b++) {
            int l = baseToLocal[b];
            if (l == fromClass) {
                l = toClass;
            }
            if (l > fromClass) {
                l--;
            }
            newMap[b] = l;
        }
        int number = Integer.parseInt(name.substring(1)) + 1;
        return new ClassPartition("A" + number, newNames, newDecisions, newMap);
    }

    /**
     * Возвращает стандартное разбиение по имени. A2 по умолчанию — A1 с исключением класса
     * ΩA1_3 «похоже на яблоко» и переносом его объектов в ΩA1_2 «явно не яблоко».
     *
     * @param name имя ("A1" или "A2", регистр не важен)
     * @return разбиение
     * @throws IllegalArgumentException для неизвестного имени
     */
    public static ClassPartition fromName(String name) {
        if (name == null) {
            throw new IllegalArgumentException("Имя разбиения не задано");
        }
        switch (name.trim().toUpperCase(Locale.ROOT)) {
            case "A1":
                return A1;
            case "A2":
                return A1.merge(2, 1);
            default:
                throw new IllegalArgumentException("Неизвестное разбиение: " + name + " (доступны A1, A2)");
        }
    }

    @Override
    public String toString() {
        return name;
    }
}
