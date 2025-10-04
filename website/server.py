#!/usr/bin/env python3
"""
QBES Website Server
Flask server to serve the website and handle testing API calls
"""

import os
import sys
import json
import threading
import time
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

# Add QBES to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_runner import QBESTestRunner

app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)

# Global test runner instance
test_runner = QBESTestRunner()

# Store running tests
running_tests = {}

@app.route('/')
def index():
    """Serve the main website."""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('.', filename)

@app.route('/api/status')
def get_status():
    """Get project status and statistics."""
    try:
        status = test_runner.get_project_status()
        return jsonify({
            'success': True,
            'data': status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test/<test_type>')
def run_test_api(test_type):
    """Run tests via API."""
    if test_type in running_tests:
        return jsonify({
            'success': False,
            'error': 'Test already running',
            'status': 'running'
        })
    
    # Start test in background thread
    test_id = f"{test_type}_{int(time.time())}"
    running_tests[test_id] = {
        'status': 'starting',
        'results': None
    }
    
    def run_test_background():
        try:
            if test_type == 'core':
                results = test_runner.run_core_tests()
            elif test_type == 'benchmarks':
                results = test_runner.run_benchmark_tests()
            elif test_type == 'validation':
                results = test_runner.run_validation_tests()
            elif test_type == 'all':
                results = test_runner.run_all_tests()
            else:
                results = {
                    'success': False,
                    'error': f'Unknown test type: {test_type}'
                }
            
            running_tests[test_id]['status'] = 'completed'
            running_tests[test_id]['results'] = results
            
        except Exception as e:
            running_tests[test_id]['status'] = 'failed'
            running_tests[test_id]['results'] = {
                'success': False,
                'error': str(e)
            }
    
    thread = threading.Thread(target=run_test_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'test_id': test_id,
        'status': 'started'
    })

@app.route('/api/test/status/<test_id>')
def get_test_status(test_id):
    """Get status of running test."""
    if test_id not in running_tests:
        return jsonify({
            'success': False,
            'error': 'Test not found'
        }), 404
    
    test_info = running_tests[test_id]
    
    return jsonify({
        'success': True,
        'status': test_info['status'],
        'results': test_info['results']
    })

@app.route('/api/demo', methods=['POST'])
def run_demo():
    """Run demo simulation with parameters."""
    try:
        data = request.get_json() or {}
        
        energy_gap = float(data.get('energy_gap', 2.0))
        coupling = float(data.get('coupling', 0.1))
        temperature = float(data.get('temperature', 300.0))
        noise_type = data.get('noise_type', 'protein')
        
        results = test_runner.run_demo_simulation(
            energy_gap=energy_gap,
            coupling=coupling,
            temperature=temperature,
            noise_type=noise_type
        )
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/validate')
def validate_installation():
    """Validate QBES installation."""
    try:
        validation_results = {
            'modules': {},
            'dependencies': {},
            'functionality': {},
            'overall_status': True
        }
        
        # Test module imports
        modules_to_test = [
            'qbes',
            'qbes.config_manager',
            'qbes.quantum_engine',
            'qbes.noise_models',
            'qbes.analysis'
        ]
        
        for module in modules_to_test:
            try:
                __import__(module)
                validation_results['modules'][module] = 'OK'
            except ImportError as e:
                validation_results['modules'][module] = f'FAILED: {str(e)}'
                validation_results['overall_status'] = False
        
        # Test dependencies
        dependencies = [
            'numpy',
            'scipy',
            'matplotlib',
            'openmm',
            'mdtraj'
        ]
        
        for dep in dependencies:
            try:
                __import__(dep)
                validation_results['dependencies'][dep] = 'OK'
            except ImportError:
                validation_results['dependencies'][dep] = 'MISSING'
                if dep in ['openmm', 'mdtraj']:
                    # Optional dependencies
                    pass
                else:
                    validation_results['overall_status'] = False
        
        # Test basic functionality
        try:
            from qbes import ConfigurationManager
            cm = ConfigurationManager()
            validation_results['functionality']['config_manager'] = 'OK'
        except Exception as e:
            validation_results['functionality']['config_manager'] = f'FAILED: {str(e)}'
            validation_results['overall_status'] = False
        
        try:
            from qbes import QuantumEngine
            qe = QuantumEngine()
            hamiltonian = qe.create_two_level_hamiltonian(2.0, 0.1)
            validation_results['functionality']['quantum_engine'] = 'OK'
        except Exception as e:
            validation_results['functionality']['quantum_engine'] = f'FAILED: {str(e)}'
            validation_results['overall_status'] = False
        
        return jsonify({
            'success': True,
            'data': validation_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/examples')
def get_examples():
    """Get example configurations and code snippets."""
    examples = {
        'configurations': {
            'photosystem': {
                'name': 'Photosynthetic Complex',
                'description': 'Light-harvesting complex simulation',
                'config': {
                    'system': {
                        'pdb_file': 'photosystem.pdb',
                        'force_field': 'amber14'
                    },
                    'simulation': {
                        'temperature': 300.0,
                        'simulation_time': 1e-12
                    },
                    'quantum_subsystem': {
                        'selection_method': 'chromophores'
                    },
                    'noise_model': {
                        'type': 'protein_ohmic',
                        'coupling_strength': 2.0
                    }
                }
            },
            'enzyme': {
                'name': 'Enzyme Active Site',
                'description': 'Quantum tunneling in enzymatic reactions',
                'config': {
                    'system': {
                        'pdb_file': 'enzyme.pdb',
                        'force_field': 'amber14'
                    },
                    'simulation': {
                        'temperature': 310.0,
                        'simulation_time': 5e-13
                    },
                    'quantum_subsystem': {
                        'selection_method': 'active_site'
                    },
                    'noise_model': {
                        'type': 'protein_ohmic',
                        'coupling_strength': 1.5
                    }
                }
            }
        },
        'code_snippets': {
            'basic_simulation': '''from qbes import ConfigurationManager, SimulationEngine

# Load configuration
config_manager = ConfigurationManager()
config = config_manager.load_config("my_simulation.yaml")

# Run simulation
engine = SimulationEngine()
engine.initialize_simulation(config)
results = engine.run_simulation()

# Analyze results
print(f"Coherence lifetime: {results.coherence_lifetime:.2e} s")''',
            
            'quantum_system': '''from qbes import QuantumEngine, NoiseModelFactory

# Create quantum system
quantum_engine = QuantumEngine()
hamiltonian = quantum_engine.create_two_level_hamiltonian(
    energy_gap=2.0,  # eV
    coupling=0.1     # eV
)

# Add biological noise
noise_factory = NoiseModelFactory()
protein_noise = noise_factory.create_protein_noise_model(
    temperature=300.0  # K
)''',
            
            'analysis': '''from qbes import ResultsAnalyzer

# Analyze simulation results
analyzer = ResultsAnalyzer()

# Calculate coherence measures
coherence_metrics = analyzer.generate_coherence_metrics(state_trajectory)

# Validate against literature
validation_result = analyzer.validate_against_theoretical_predictions(
    measured_values, theoretical_values
)'''
        }
    }
    
    return jsonify({
        'success': True,
        'data': examples
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

def main():
    """Main function to run the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='QBES Website Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("QBES Website Server")
    print("=" * 60)
    print(f"Starting server on http://{args.host}:{args.port}")
    print("Available endpoints:")
    print("  GET  /                    - Main website")
    print("  GET  /api/status          - Project status")
    print("  GET  /api/test/<type>     - Run tests")
    print("  POST /api/demo            - Run demo simulation")
    print("  GET  /api/validate        - Validate installation")
    print("  GET  /api/examples        - Get examples")
    print("=" * 60)
    
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == '__main__':
    main()