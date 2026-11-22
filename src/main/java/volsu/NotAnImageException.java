// src/main/java/volsu/NotAnImageException.java
package volsu;

/**
 * Исключение: файл существует, но не является читаемым изображением
 * (формат не поддерживается {@code ImageIO} либо файл повреждён).
 * В CLI приводит к коду возврата {@code 1} (ошибка ввода).
 */
public class NotAnImageException extends Exception {

    private static final long serialVersionUID = 1L;

    /**
     * Создаёт исключение с сообщением.
     *
     * @param message описание проблемы
     */
    public NotAnImageException(String message) {
        super(message);
    }

    /**
     * Создаёт исключение с сообщением и причиной.
     *
     * @param message описание проблемы
     * @param cause   исходная причина
     */
    public NotAnImageException(String message, Throwable cause) {
        super(message, cause);
    }
}
