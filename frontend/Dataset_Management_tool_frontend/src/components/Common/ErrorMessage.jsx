import { FiAlertCircle, FiX } from 'react-icons/fi';
import './ErrorMessage.css';

function ErrorMessage({ message, onClose }) {
    if (!message) return null;

    return (
        <div className="error-message glass">
            <FiAlertCircle className="error-icon" />
            <p className="error-text">{message}</p>
            {onClose && (
                <button className="error-close" onClick={onClose}>
                    <FiX />
                </button>
            )}
        </div>
    );
}

export default ErrorMessage;
