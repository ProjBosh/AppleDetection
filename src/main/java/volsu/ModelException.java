// src/main/java/volsu/ModelException.java
package volsu;

/**
 * Исключение: модель распознавания отсутствует, повреждена или некорректна.
 * В CLI приводит к коду возврата {@code 2} (ошибка модели).
 */
public class ModelException extends Exception {

    private static final long serialVersionUID = 1L;

    /**
     * Создаёт исключение с сообщением.
     *
     * @param message описание проблемы
     */
    public ModelException(String message) {
        super(message);
    }

    /**
     * Создаёт исключение с сообщением и причиной.
     *
     * @param message описание проблемы
     * @param cause   исходная причина
     */
    public ModelException(String message, Throwable cause) {
        super(message, cause);
    }
}
