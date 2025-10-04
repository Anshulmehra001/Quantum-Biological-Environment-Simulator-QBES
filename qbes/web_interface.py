#!/usr/bin/env python3
"""
QBES Web Interface - User-friendly web interface for running QBES simulations
"""

import os
import sys
import json
import threading
import time
from flask import Flask, render_template_string, request, jsonify, send_from_directory
from flask_cors import CORS

# Add QBES to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from qbes import ConfigurationManager, QuantumEngine, NoiseModelFactory, ResultsAnalyzer
    QBES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: QBES modules not fully available: {e}")
    QBES_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Store running simulations
running_simulations = {}

@app.route('/')
def index():
    """Main QBES web interface."""
    return render_template_string(MAIN_TEMPLATE)

@app.route('/api/status')
def get_status():
    """Get QBES system status."""
    status = {
        'qbes_available': QBES_AVAILABLE,
        'version': '0.1.0',
        'running_simulations': len(running_simulations)
    }
    return jsonify(status)

@app.route('/api/create_simulation', methods=['POST'])
def create_simulation():
    """Create a new simulation configuration."""
    try:
        data = request.get_json()
        
        # Extract parameters
        sim_name = data.get('name', 'my_simulation')
        system_type = data.get('system_type', 'photosystem')
        temperature = float(data.get('temperature', 300.0))
        simulation_time = float(data.get('simulation_time', 1e-12))
        
        if not QBES_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'QBES modules not available'
            })
        
        # Create configuration
        config_manager = ConfigurationManager()
        config_file = f"{sim_name}.yaml"
        
        # Generate configuration based on system type
        if system_type == 'photosystem':
            success = config_manager.generate_default_config(config_file)
        else:
            success = config_manager.generate_default_config(config_file)
        
        if success:
            return jsonify({
                'success': True,
                'config_file': config_file,
                'message': f'Configuration created for {system_type} simulation'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to create configuration'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# HTML Template for the web interface
MAIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QBES - Quantum Biology Simulator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        .main-content {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
        }
        .section h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5rem;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 1rem;
        }
        .btn {
            background: #667eea;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.3s;
        }
        .btn:hover {
            background: #5a67d8;
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .status {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 QBES Interface</h1>
            <p>Quantum Biological Environment Simulator</p>
        </div>
        
        <div class="main-content">
            <!-- System Status -->
            <div class="section">
                <h2>📊 System Status</h2>
                <div id="system-status">
                    <p>Checking QBES availability...</p>
                </div>
            </div>
            
            <!-- Quick Start -->
            <div class="section">
                <h2>🚀 Quick Start Simulation</h2>
                <div class="grid">
                    <div>
                        <div class="form-group">
                            <label for="sim-name">Simulation Name:</label>
                            <input type="text" id="sim-name" value="my_quantum_simulation" placeholder="Enter simulation name">
                        </div>
                        
                        <div class="form-group">
                            <label for="system-type">Biological System:</label>
                            <select id="system-type">
                                <option value="photosystem">Photosynthetic Complex</option>
                                <option value="enzyme">Enzyme Active Site</option>
                                <option value="membrane">Membrane Protein</option>
                                <option value="dna">DNA/RNA System</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="temperature">Temperature (K):</label>
                            <input type="number" id="temperature" value="300" min="77" max="400">
                        </div>
                        
                        <div class="form-group">
                            <label for="sim-time">Simulation Time (ps):</label>
                            <input type="number" id="sim-time" value="1.0" min="0.1" max="100" step="0.1">
                        </div>
                    </div>
                    
                    <div>
                        <h3>What This Will Do:</h3>
                        <div id="system-description">
                            <p><strong>Photosynthetic Complex:</strong> Simulates quantum coherence in light-harvesting complexes, showing how plants efficiently capture and transfer solar energy.</p>
                        </div>
                        
                        <h3>Expected Results:</h3>
                        <ul>
                            <li>Quantum coherence lifetime</li>
                            <li>Energy transfer efficiency</li>
                            <li>Decoherence rates</li>
                            <li>Environmental effects</li>
                        </ul>
                    </div>
                </div>
                
                <button class="btn" onclick="createSimulation()">
                    🧪 Create Simulation Configuration
                </button>
                
                <div id="simulation-result"></div>
            </div>
            
            <!-- Available Tools -->
            <div class="section">
                <h2>🛠️ Available Tools</h2>
                <div class="grid">
                    <div>
                        <h3>Configuration Tools</h3>
                        <button class="btn" onclick="generateConfig()">Generate Config File</button>
                        <button class="btn" onclick="validateConfig()">Validate Configuration</button>
                    </div>
                    <div>
                        <h3>Analysis Tools</h3>
                        <button class="btn" onclick="runBenchmarks()">Run Benchmarks</button>
                        <button class="btn" onclick="runDemo()">Run Demo</button>
                    </div>
                </div>
            </div>
            
            <!-- Help & Documentation -->
            <div class="section">
                <h2>📚 Help & Documentation</h2>
                <p>New to QBES? Here are some resources to get you started:</p>
                <ul>
                    <li><strong>What is QBES?</strong> - Read WHAT_IS_QBES.md for a simple explanation</li>
                    <li><strong>Tutorial Website</strong> - Open website/qbes_website.html for interactive learning</li>
                    <li><strong>User Guide</strong> - Check HOW_TO_USE_QBES.md for detailed instructions</li>
                    <li><strong>API Documentation</strong> - See docs/ folder for technical details</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        // System descriptions for different biological systems
        const systemDescriptions = {
            photosystem: "Simulates quantum coherence in light-harvesting complexes, showing how plants efficiently capture and transfer solar energy.",
            enzyme: "Models quantum tunneling effects in enzyme active sites, explaining how biological catalysts achieve remarkable efficiency.",
            membrane: "Studies quantum effects in membrane proteins like ion channels, revealing quantum mechanisms in cellular transport.",
            dna: "Investigates quantum effects in DNA/RNA systems, exploring quantum contributions to genetic processes."
        };
        
        // Update system description when selection changes
        document.getElementById('system-type').addEventListener('change', function() {
            const description = systemDescriptions[this.value];
            document.getElementById('system-description').innerHTML = 
                `<p><strong>${this.options[this.selectedIndex].text}:</strong> ${description}</p>`;
        });
        
        // Check system status on load
        window.addEventListener('load', function() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const statusDiv = document.getElementById('system-status');
                    if (data.qbes_available) {
                        statusDiv.innerHTML = `
                            <div class="status success">
                                ✅ QBES is available and ready to use!<br>
                                Version: ${data.version}<br>
                                Running simulations: ${data.running_simulations}
                            </div>
                        `;
                    } else {
                        statusDiv.innerHTML = `
                            <div class="status error">
                                ❌ QBES modules not fully available. Some features may be limited.
                            </div>
                        `;
                    }
                })
                .catch(error => {
                    document.getElementById('system-status').innerHTML = `
                        <div class="status error">
                            ⚠️ Could not check system status. Running in offline mode.
                        </div>
                    `;
                });
        });
        
        // Create simulation configuration
        function createSimulation() {
            const data = {
                name: document.getElementById('sim-name').value,
                system_type: document.getElementById('system-type').value,
                temperature: document.getElementById('temperature').value,
                simulation_time: parseFloat(document.getElementById('sim-time').value) * 1e-12
            };
            
            fetch('/api/create_simulation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                const resultDiv = document.getElementById('simulation-result');
                if (result.success) {
                    resultDiv.innerHTML = `
                        <div class="status success">
                            ✅ ${result.message}<br>
                            Configuration file: ${result.config_file}<br>
                            <strong>Next steps:</strong>
                            <ol>
                                <li>Edit the configuration file if needed</li>
                                <li>Run: python -m qbes.cli validate ${result.config_file}</li>
                                <li>Run: python -m qbes.cli run ${result.config_file}</li>
                            </ol>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <div class="status error">
                            ❌ Error: ${result.error}
                        </div>
                    `;
                }
            })
            .catch(error => {
                document.getElementById('simulation-result').innerHTML = `
                    <div class="status error">
                        ❌ Network error: ${error.message}
                    </div>
                `;
            });
        }
        
        // Placeholder functions for other tools
        function generateConfig() {
            alert('This will open the configuration generator. For now, use the Quick Start above or run: python -m qbes.cli generate-config');
        }
        
        function validateConfig() {
            alert('This will validate your configuration. For now, run: python -m qbes.cli validate your_config.yaml');
        }
        
        function runBenchmarks() {
            alert('This will run the benchmark suite. For now, run: python run_benchmarks.py');
        }
        
        function runDemo() {
            alert('This will run the QBES demo. For now, run: python demo_qbes.py');
        }
    </script>
</body>
</html>
'''

def main():
    """Start the QBES web interface."""
    print("🚀 Starting QBES Web Interface...")
    print("📍 Open your browser to: http://localhost:8080")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print("\n👋 QBES Web Interface stopped")

if __name__ == '__main__':
    main()