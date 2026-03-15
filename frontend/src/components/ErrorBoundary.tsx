import { Component, ReactNode } from 'react';
import { Link } from 'react-router-dom';

interface Props { children: ReactNode }
interface State { hasError: boolean }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error) {
    console.error('ErrorBoundary:', error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="max-w-2xl mx-auto px-8 py-24 text-center">
          <h2 className="text-xl font-bold text-red-400 mb-3">Something went wrong</h2>
          <Link
            to="/"
            onClick={() => this.setState({ hasError: false })}
            className="text-indigo-400 hover:text-indigo-300"
          >
            ← Return to dashboard
          </Link>
        </div>
      );
    }
    return this.props.children;
  }
}
