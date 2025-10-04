#!/usr/bin/env python3
"""
QBES Web Application
Full-featured web interface for quantum biology simulations
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# Add QBES to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from qbes.config_manager import ConfigManager
    from qbes.simulation_engine import SimulationEngine
    from qbes.analysis import AnalysisEngine
    from qbes.visualization import VisualizationEngine
    from qbes.core.data_models import SimulationConfig, SystemType, NoiseModel
    QBES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: QBES modules not available: {e}")
    QBES_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'qbes-web-interface-2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'web_results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

class QBESWebInterface:
    """Main web interface class for QBES"""
    
    def __init__(self):
        self.results_cache = {}
        self.simulation_history = []
        
    def create_default_config(self, system_type="two_level", **params):
        """Create a default simulation configuration"""
        if not QBES_AVAILABLE:
            return self._mock_config(system_type, **params)
            
        config = {
            "system": {
                "type": system_type,
                "temperature": params.get("temperature", 300.0),
                "hamiltonian": {
                    "energy_gap": params.get("energy_gap", 2.0),
                    "coupling": params.get("coupling", 0.1)
                }
            },
            "simulation": {
                "time_step": 0.1,
                "total_time": params.get("total_time", 10.0),
                "method": "lindblad"
            },
            "noise": {
                "model": params.get("noise_model", "protein_ohmic"),
                "strength": params.get("noise_strength", 0.1)
            },
            "output": {
                "save_trajectory": True,
                "analysis": ["coherence", "purity", "energy"]
            }
        }
        return config
    
    def _mock_config(self, system_type, **params):
        """Mock configuration when QBES is not available"""
        return {
            "system_type": system_type,
            "temperature": params.get("temperature", 300.0),
            "energy_gap": params.get("energy_gap", 2.0),
            "coupling": params.get("coupling", 0.1),
            "noise_model": params.get("noise_model", "protein_ohmic"),
            "total_time": params.get("total_time", 10.0)
        }
    
    def run_simulation(self, config):
        """Run a QBES simulation"""
        if not QBES_AVAILABLE:
            return self._mock_simulation(config)
            
        try:
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f, indent=2)
                config_file = f.name
            
            # Run simulation
            config_manager = ConfigManager()
            sim_config = config_manager.load_config(config_file)
            
            engine = SimulationEngine()
            results = engine.run_simulation(sim_config)
            
            # Clean up
            os.unlink(config_file)
            
            return self._process_results(results)
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _mock_simulation(self, config):
        """Mock simulation results when QBES is not available"""
        # Generate realistic mock data
        time_points = np.linspace(0, config.get("total_time", 10.0), 101)
        
        # Calculate decoherence based on parameters
        temp = config.get("temperature", 300.0)
        noise_strength = 0.1 if config.get("noise_model") == "protein_ohmic" else 0.05
        decoherence_rate = noise_strength * (temp / 300.0) ** 0.5
        
        coherence = np.exp(-decoherence_rate * time_points)
        purity = 0.5 + 0.5 * coherence
        energy = np.full_like(time_points, config.get("energy_gap", 2.0) / 2)
        
        return {
            "success": True,
            "time_points": time_points.tolist(),
            "coherence": coherence.tolist(),
            "purity": purity.tolist(),
            "energy": energy.tolist(),
            "decoherence_rate": decoherence_rate,
            "coherence_lifetime": 1.0 / decoherence_rate if decoherence_rate > 0 else float('inf'),
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
    
    def _process_results(self, results):
        """Process simulation results for web display"""
        processed = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "config": results.get("config", {}),
            "time_points": results.get("time_evolution", {}).get("time", []),
            "coherence": results.get("analysis", {}).get("coherence", []),
            "purity": results.get("analysis", {}).get("purity", []),
            "energy": results.get("time_evolution", {}).get("energy", [])
        }
        
        # Calculate derived quantities
        if processed["coherence"]:
            coherence_array = np.array(processed["coherence"])
            time_array = np.array(processed["time_points"])
            
            # Fit exponential decay
            if len(coherence_array) > 1:
                try:
                    # Simple exponential fit
                    log_coherence = np.log(np.maximum(coherence_array, 1e-10))
                    fit = np.polyfit(time_array, log_coherence, 1)
                    decoherence_rate = -fit[0]
                    processed["decoherence_rate"] = max(0, decoherence_rate)
                    processed["coherence_lifetime"] = 1.0 / decoherence_rate if decoherence_rate > 0 else float('inf')
                except:
                    processed["decoherence_rate"] = 0.1
                    processed["coherence_lifetime"] = 10.0
        
        return processed
    
    def create_plot(self, results, plot_type="coherence"):
        """Create matplotlib plot and return as base64 string"""
        plt.figure(figsize=(10, 6))
        
        time_points = results.get("time_points", [])
        
        if plot_type == "coherence":
            data = results.get("coherence", [])
            plt.plot(time_points, data, 'b-', linewidth=2, label='Quantum Coherence')
            plt.ylabel('Coherence')
            plt.title('Quantum Coherence Evolution')
            
        elif plot_type == "purity":
            data = results.get("purity", [])
            plt.plot(time_points, data, 'r-', linewidth=2, label='State Purity')
            plt.ylabel('Purity')
            plt.title('Quantum State Purity Evolution')
            
        elif plot_type == "energy":
            data = results.get("energy", [])
            plt.plot(time_points, data, 'g-', linewidth=2, label='Energy')
            plt.ylabel('Energy (eV)')
            plt.title('System Energy Evolution')
            
        elif plot_type == "all":
            coherence = results.get("coherence", [])
            purity = results.get("purity", [])
            
            plt.subplot(2, 1, 1)
            plt.plot(time_points, coherence, 'b-', linewidth=2, label='Coherence')
            plt.ylabel('Coherence')
            plt.title('Quantum Properties Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(time_points, purity, 'r-', linewidth=2, label='Purity')
            plt.ylabel('Purity')
            plt.xlabel('Time (ps)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        if plot_type != "all":
            plt.xlabel('Time (ps)')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64

# Initialize web interface
web_interface = QBESWebInterface()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/simulator')
def simulator():
    """Simulation interface page"""
    return render_template('simulator.html')

@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    """API endpoint for running simulations"""
    try:
        data = request.get_json()
        
        # Extract parameters
        config = web_interface.create_default_config(
            system_type=data.get('system_type', 'two_level'),
            temperature=float(data.get('temperature', 300)),
            energy_gap=float(data.get('energy_gap', 2.0)),
            coupling=float(data.get('coupling', 0.1)),
            noise_model=data.get('noise_model', 'protein_ohmic'),
            total_time=float(data.get('total_time', 10.0)),
            noise_strength=float(data.get('noise_strength', 0.1))
        )
        
        # Run simulation
        results = web_interface.run_simulation(config)
        
        if results.get("success"):
            # Store in history
            web_interface.simulation_history.append(results)
            
            # Generate plots
            plots = {}
            for plot_type in ['coherence', 'purity', 'all']:
                try:
                    plots[plot_type] = web_interface.create_plot(results, plot_type)
                except Exception as e:
                    print(f"Plot generation error for {plot_type}: {e}")
                    plots[plot_type] = None
            
            results['plots'] = plots
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/presets')
def api_presets():
    """Get simulation presets"""
    presets = {
        "photosynthesis": {
            "name": "Photosynthetic Complex",
            "system_type": "two_level",
            "temperature": 300,
            "energy_gap": 1.5,
            "coupling": 0.05,
            "noise_model": "protein_ohmic",
            "total_time": 20.0,
            "description": "Models energy transfer in photosynthetic light-harvesting complexes"
        },
        "enzyme": {
            "name": "Enzyme Active Site",
            "system_type": "two_level",
            "temperature": 310,
            "energy_gap": 2.5,
            "coupling": 0.2,
            "noise_model": "protein_ohmic",
            "total_time": 5.0,
            "description": "Models quantum tunneling in enzyme catalysis"
        },
        "membrane": {
            "name": "Membrane Protein",
            "system_type": "two_level",
            "temperature": 300,
            "energy_gap": 1.0,
            "coupling": 0.1,
            "noise_model": "membrane_fluctuations",
            "total_time": 15.0,
            "description": "Models quantum effects in membrane-bound proteins"
        },
        "cold_system": {
            "name": "Cryogenic System",
            "system_type": "two_level",
            "temperature": 77,
            "energy_gap": 3.0,
            "coupling": 0.01,
            "noise_model": "protein_ohmic",
            "total_time": 50.0,
            "description": "Low-temperature system with long coherence times"
        }
    }
    return jsonify(presets)

@app.route('/api/history')
def api_history():
    """Get simulation history"""
    return jsonify({
        "history": web_interface.simulation_history[-10:],  # Last 10 simulations
        "total_count": len(web_interface.simulation_history)
    })

@app.route('/api/export/<format>')
def api_export(format):
    """Export simulation results"""
    if not web_interface.simulation_history:
        return jsonify({"error": "No simulation results to export"}), 400
    
    latest_results = web_interface.simulation_history[-1]
    
    if format == 'json':
        # Create temporary file
        filename = f"qbes_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        
        with open(filepath, 'w') as f:
            json.dump(latest_results, f, indent=2)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    
    elif format == 'csv':
        # Create CSV file
        filename = f"qbes_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        
        import pandas as pd
        df = pd.DataFrame({
            'Time_ps': latest_results.get('time_points', []),
            'Coherence': latest_results.get('coherence', []),
            'Purity': latest_results.get('purity', []),
            'Energy_eV': latest_results.get('energy', [])
        })
        df.to_csv(filepath, index=False)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    
    else:
        return jsonify({"error": "Unsupported format"}), 400

@app.route('/api/system_info')
def api_system_info():
    """Get system information"""
    info = {
        "qbes_available": QBES_AVAILABLE,
        "python_version": sys.version,
        "platform": sys.platform,
        "simulation_count": len(web_interface.simulation_history),
        "available_models": [
            "protein_ohmic",
            "membrane_fluctuations", 
            "solvent_dynamics",
            "thermal_bath"
        ],
        "system_types": [
            "two_level",
            "three_level",
            "spin_chain"
        ]
    }
    return jsonify(info)

if __name__ == '__main__':
    print("🚀 Starting QBES Web Application...")
    print(f"📊 QBES Backend Available: {QBES_AVAILABLE}")
    print("🌐 Open your browser to: http://localhost:5000")
    print("📱 Mobile-friendly interface included")
    print("⚡ Real-time simulation capabilities")
    
    app.run(debug=True, host='0.0.0.0', port=5000)