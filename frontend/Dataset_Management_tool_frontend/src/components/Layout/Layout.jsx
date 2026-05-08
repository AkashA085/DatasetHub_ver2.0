import { Link, useLocation } from 'react-router-dom';
import { FiHome, FiDatabase, FiUpload, FiZap, FiCpu, FiActivity } from 'react-icons/fi';
import './Layout.css';

function Layout({ children }) {
    const location = useLocation();

    const navItems = [
        { path: '/', icon: FiHome, label: 'Dashboard' },
        { path: '/datasets', icon: FiDatabase, label: 'Datasets' },
        { path: '/upload', icon: FiUpload, label: 'Upload' },
        { path: '/augment', icon: FiZap, label: 'Augment' },
        { path: '/train', icon: FiActivity, label: 'Train' },
        { path: '/models', icon: FiCpu, label: 'Models' },
    ];

    return (
        <div className="layout">
            <aside className="sidebar glass">
                <div className="sidebar-header">
                    <h1 className="logo">
                        <span className="logo-icon">DH</span>
                        <span className="logo-text">DatasetHub</span>
                    </h1>
                </div>

                <nav className="sidebar-nav">
                    {navItems.map((item) => {
                        const Icon = item.icon;
                        const isActive = location.pathname === item.path;

                        return (
                            <Link
                                key={item.path}
                                to={item.path}
                                className={`nav-item ${isActive ? 'active' : ''}`}
                            >
                                <Icon className="nav-icon" />
                                <span className="nav-label">{item.label}</span>
                            </Link>
                        );
                    })}
                </nav>

                <div className="sidebar-footer">
                    <p className="text-sm text-secondary">
                        Dataset Management v1.0
                    </p>
                </div>
            </aside>

            <main className="main-content">
                <div className="content-wrapper">
                    {children}
                </div>
            </main>
        </div>
    );
}

export default Layout;
