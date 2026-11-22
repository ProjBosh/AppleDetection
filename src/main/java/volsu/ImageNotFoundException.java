// src/main/java/volsu/ImageNotFoundException.java
package volsu;

/**
 * Исключение: файл изображения не найден (или это не обычный файл).
 * В CLI приводит к коду возврата {@code 1} (ошибка ввода).
 */
public class ImageNotFoundException extends Exception {

    private static final long serialVersionUID = 1L;

    /**
     * Создаёт исключение с сообщением.
     *
     * @param message описание проблемы
     */
    public ImageNotFoundException(String message) {
        super(message);
    }
}
