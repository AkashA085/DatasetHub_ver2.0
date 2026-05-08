import React from 'react';

class AppErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error('App runtime error:', error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{ padding: '24px', maxWidth: '980px', margin: '0 auto' }}>
                    <h2 style={{ marginBottom: '10px' }}>Runtime error in page</h2>
                    <p style={{ marginBottom: '12px' }}>
                        The app crashed while rendering this route. Please share the error below.
                    </p>
                    <pre style={{
                        background: '#0f1726',
                        color: '#cce0ff',
                        borderRadius: '10px',
                        padding: '12px',
                        overflow: 'auto',
                    }}>
                        {this.state.error?.stack || this.state.error?.message || 'Unknown error'}
                    </pre>
                </div>
            );
        }

        return this.props.children;
    }
}

export default AppErrorBoundary;
