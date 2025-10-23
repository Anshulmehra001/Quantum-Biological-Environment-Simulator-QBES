#!/usr/bin/env python3
"""
Create demo PDB files for QBES simulations
"""

def create_photosystem_pdb():
    """Create a simple but properly formatted photosystem PDB file"""
    # This creates a minimal valid peptide structure (ALA-GLY-VAL-PHE)
    # without explicit CONECT records - OpenMM will infer bonds from standard residues
    pdb_content = """HEADER    PHOTOSYSTEM                             01-JAN-25   DEMO            
TITLE     DEMO PHOTOSYSTEM FOR QBES SIMULATION                                   
REMARK   1 SIMPLE 4-RESIDUE PEPTIDE FOR QBES TESTING
ATOM      1  N   ALA A   1      20.154  16.967  27.462  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  27.462  1.00 20.00           C  
ATOM      3  C   ALA A   1      17.693  16.849  27.462  1.00 20.00           C  
ATOM      4  O   ALA A   1      17.693  18.081  27.462  1.00 20.00           O  
ATOM      5  CB  ALA A   1      19.030  15.235  26.196  1.00 20.00           C  
ATOM      6  N   GLY A   2      16.632  16.101  27.462  1.00 20.00           N  
ATOM      7  CA  GLY A   2      15.219  16.849  27.462  1.00 20.00           C  
ATOM      8  C   GLY A   2      14.095  15.983  27.462  1.00 20.00           C  
ATOM      9  O   GLY A   2      14.095  14.751  27.462  1.00 20.00           O  
ATOM     10  N   VAL A   3      12.971  16.683  27.462  1.00 20.00           N  
ATOM     11  CA  VAL A   3      11.684  15.983  27.462  1.00 20.00           C  
ATOM     12  C   VAL A   3      10.560  16.849  27.462  1.00 20.00           C  
ATOM     13  O   VAL A   3      10.560  18.081  27.462  1.00 20.00           O  
ATOM     14  CB  VAL A   3      11.684  15.117  26.196  1.00 20.00           C  
ATOM     15  CG1 VAL A   3      10.560  14.251  26.196  1.00 20.00           C  
ATOM     16  CG2 VAL A   3      12.808  14.251  26.196  1.00 20.00           C  
ATOM     17  N   PHE A   4       9.437  16.149  27.462  1.00 20.00           N  
ATOM     18  CA  PHE A   4       8.150  16.849  27.462  1.00 20.00           C  
ATOM     19  C   PHE A   4       7.026  15.983  27.462  1.00 20.00           C  
ATOM     20  O   PHE A   4       7.026  14.751  27.462  1.00 20.00           O  
ATOM     21  OXT PHE A   4       5.965  16.621  27.462  1.00 20.00           O  
ATOM     22  CB  PHE A   4       8.150  17.715  28.728  1.00 20.00           C  
ATOM     23  CG  PHE A   4       9.274  18.581  28.728  1.00 20.00           C  
ATOM     24  CD1 PHE A   4      10.398  18.581  29.541  1.00 20.00           C  
ATOM     25  CD2 PHE A   4       9.274  19.447  27.645  1.00 20.00           C  
ATOM     26  CE1 PHE A   4      11.459  19.447  29.541  1.00 20.00           C  
ATOM     27  CE2 PHE A   4      10.335  20.313  27.645  1.00 20.00           C  
ATOM     28  CZ  PHE A   4      11.459  20.313  28.458  1.00 20.00           C  
TER      29      PHE A   4
END                                                                             
"""
    
    with open("photosystem.pdb", "w") as f:
        f.write(pdb_content)
    print("✅ Created photosystem.pdb")

def create_enzyme_pdb():
    """Create a simple enzyme PDB file"""
    pdb_content = """HEADER    ENZYME ACTIVE SITE                      01-JAN-25   DEMO            
TITLE     DEMO ENZYME FOR QBES SIMULATION                                       
ATOM      1  N   HIS A   1      20.154  16.967  27.462  1.00 20.00           N  
ATOM      2  CA  HIS A   1      19.030  16.101  27.462  1.00 20.00           C  
ATOM      3  C   HIS A   1      17.693  16.849  27.462  1.00 20.00           C  
ATOM      4  O   HIS A   1      16.632  16.329  27.462  1.00 20.00           O  
ATOM      5  CB  HIS A   1      19.030  15.235  26.196  1.00 20.00           C  
ATOM      6  CG  HIS A   1      17.793  14.487  26.196  1.00 20.00           C  
ATOM      7  ND1 HIS A   1      16.556  14.487  25.930  1.00 20.00           N  
ATOM      8  CD2 HIS A   1      17.793  13.621  25.930  1.00 20.00           C  
ATOM      9  CE1 HIS A   1      15.819  13.621  25.930  1.00 20.00           C  
ATOM     10  NE2 HIS A   1      16.556  12.873  25.930  1.00 20.00           N  
ATOM     11  N   CYS A   2      17.693  18.115  27.462  1.00 20.00           N  
ATOM     12  CA  CYS A   2      16.456  18.863  27.462  1.00 20.00           C  
ATOM     13  C   CYS A   2      15.219  18.115  27.462  1.00 20.00           C  
ATOM     14  O   CYS A   2      14.158  18.635  27.462  1.00 20.00           O  
ATOM     15  CB  CYS A   2      16.456  19.729  28.728  1.00 20.00           C  
ATOM     16  SG  CYS A   2      15.219  20.477  28.728  1.00 20.00           S  
ATOM     17  N   ASP A   3      15.219  16.849  27.462  1.00 20.00           N  
ATOM     18  CA  ASP A   3      13.982  16.101  27.462  1.00 20.00           C  
ATOM     19  C   ASP A   3      12.745  16.849  27.462  1.00 20.00           C  
ATOM     20  O   ASP A   3      11.684  16.329  27.462  1.00 20.00           O  
ATOM     21  CB  ASP A   3      13.982  15.235  26.196  1.00 20.00           C  
ATOM     22  CG  ASP A   3      12.745  14.487  26.196  1.00 20.00           C  
ATOM     23  OD1 ASP A   3      11.684  14.487  26.196  1.00 20.00           O  
ATOM     24  OD2 ASP A   3      12.745  13.621  25.930  1.00 20.00           O  
CONECT    1    2                                                                
CONECT    2    1    3    5                                                      
CONECT    3    2    4   11                                                      
CONECT    4    3                                                                
CONECT    5    2    6                                                           
CONECT    6    5    7    8                                                      
CONECT    7    6    9                                                           
CONECT    8    6   10                                                           
CONECT    9    7   10                                                           
CONECT   10    8    9                                                           
CONECT   11    3   12                                                           
CONECT   12   11   13   15                                                      
CONECT   13   12   14   17                                                      
CONECT   14   13                                                                
CONECT   15   12   16                                                           
CONECT   16   15                                                                
CONECT   17   13   18                                                           
CONECT   18   17   19   21                                                      
CONECT   19   18   20                                                           
CONECT   20   19                                                                
CONECT   21   18   22                                                           
CONECT   22   21   23   24                                                      
CONECT   23   22                                                                
CONECT   24   22                                                                
END                                                                             
"""
    
    with open("enzyme.pdb", "w") as f:
        f.write(pdb_content)
    print("✅ Created enzyme.pdb")

def create_membrane_pdb():
    """Create a simple membrane protein PDB file"""
    pdb_content = """HEADER    MEMBRANE PROTEIN                        01-JAN-25   DEMO            
TITLE     DEMO MEMBRANE PROTEIN FOR QBES SIMULATION                             
ATOM      1  N   LEU A   1      20.154  16.967  27.462  1.00 20.00           N  
ATOM      2  CA  LEU A   1      19.030  16.101  27.462  1.00 20.00           C  
ATOM      3  C   LEU A   1      17.693  16.849  27.462  1.00 20.00           C  
ATOM      4  O   LEU A   1      16.632  16.329  27.462  1.00 20.00           O  
ATOM      5  CB  LEU A   1      19.030  15.235  26.196  1.00 20.00           C  
ATOM      6  CG  LEU A   1      17.793  14.487  26.196  1.00 20.00           C  
ATOM      7  CD1 LEU A   1      16.556  15.235  26.196  1.00 20.00           C  
ATOM      8  CD2 LEU A   1      17.793  13.621  24.930  1.00 20.00           C  
ATOM      9  N   TRP A   2      17.693  18.115  27.462  1.00 20.00           N  
ATOM     10  CA  TRP A   2      16.456  18.863  27.462  1.00 20.00           C  
ATOM     11  C   TRP A   2      15.219  18.115  27.462  1.00 20.00           C  
ATOM     12  O   TRP A   2      14.158  18.635  27.462  1.00 20.00           O  
ATOM     13  CB  TRP A   2      16.456  19.729  28.728  1.00 20.00           C  
ATOM     14  CG  TRP A   2      15.219  20.477  28.728  1.00 20.00           C  
ATOM     15  CD1 TRP A   2      14.158  20.477  29.994  1.00 20.00           C  
ATOM     16  CD2 TRP A   2      14.158  21.343  27.462  1.00 20.00           C  
ATOM     17  NE1 TRP A   2      12.921  21.225  29.994  1.00 20.00           N  
ATOM     18  CE2 TRP A   2      12.921  22.091  28.728  1.00 20.00           C  
ATOM     19  CE3 TRP A   2      14.158  21.343  26.196  1.00 20.00           C  
ATOM     20  CZ2 TRP A   2      11.684  22.957  28.728  1.00 20.00           C  
ATOM     21  CZ3 TRP A   2      12.921  22.091  25.930  1.00 20.00           C  
ATOM     22  CH2 TRP A   2      11.684  22.957  27.462  1.00 20.00           C  
CONECT    1    2                                                                
CONECT    2    1    3    5                                                      
CONECT    3    2    4    9                                                      
CONECT    4    3                                                                
CONECT    5    2    6                                                           
CONECT    6    5    7    8                                                      
CONECT    7    6                                                                
CONECT    8    6                                                                
CONECT    9    3   10                                                           
CONECT   10    9   11   13                                                      
CONECT   11   10   12                                                           
CONECT   12   11                                                                
CONECT   13   10   14                                                           
CONECT   14   13   15   16                                                      
CONECT   15   14   17                                                           
CONECT   16   14   18   19                                                      
CONECT   17   15   18                                                           
CONECT   18   16   17   20                                                      
CONECT   19   16   21                                                           
CONECT   20   18   22                                                           
CONECT   21   19   22                                                           
CONECT   22   20   21                                                           
END                                                                             
"""
    
    with open("membrane.pdb", "w") as f:
        f.write(pdb_content)
    print("✅ Created membrane.pdb")

def create_default_pdb():
    """Create a simple default PDB file"""
    pdb_content = """HEADER    SIMPLE SYSTEM                           01-JAN-25   DEMO            
TITLE     DEMO SYSTEM FOR QBES SIMULATION                                       
ATOM      1  N   ALA A   1      20.154  16.967  27.462  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  27.462  1.00 20.00           C  
ATOM      3  C   ALA A   1      17.693  16.849  27.462  1.00 20.00           C  
ATOM      4  O   ALA A   1      16.632  16.329  27.462  1.00 20.00           O  
ATOM      5  CB  ALA A   1      19.030  15.235  26.196  1.00 20.00           C  
ATOM      6  N   GLY A   2      17.693  18.115  27.462  1.00 20.00           N  
ATOM      7  CA  GLY A   2      16.456  18.863  27.462  1.00 20.00           C  
ATOM      8  C   GLY A   2      15.219  18.115  27.462  1.00 20.00           C  
ATOM      9  O   GLY A   2      14.158  18.635  27.462  1.00 20.00           O  
CONECT    1    2                                                                
CONECT    2    1    3    5                                                      
CONECT    3    2    4    6                                                      
CONECT    4    3                                                                
CONECT    5    2                                                                
CONECT    6    3    7                                                           
CONECT    7    6    8                                                           
CONECT    8    7    9                                                           
CONECT    9    8                                                                
END                                                                             
"""
    
    with open("default.pdb", "w") as f:
        f.write(pdb_content)
    print("✅ Created default.pdb")

if __name__ == "__main__":
    print("🧬 Creating demo PDB files for QBES...")
    create_photosystem_pdb()
    create_enzyme_pdb() 
    create_membrane_pdb()
    create_default_pdb()
    print("✅ All demo PDB files created successfully!")
    print("📁 Files: photosystem.pdb, enzyme.pdb, membrane.pdb, default.pdb")