"""
Data collection and processing for Materials Project dataset.

This module provides functionality to download and process materials data from
the Materials Project API for use in the scientific visualization system.
"""
import os
import json
import logging
import time
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class MaterialsProjectDataset:
    """
    Dataset class for Materials Project data.
    
    This class handles downloading, processing, and preparing data from the
    Materials Project API for use in the scientific visualization system.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        data_dir: Optional[str] = None,
        cache: bool = True
    ):
        """
        Initialize the Materials Project dataset.
        
        Args:
            api_key: Materials Project API key (if None, will look in config and env vars)
            data_dir: Directory to store downloaded data (if None, will use config)
            cache: Whether to cache data locally
        """
        # Import here to avoid circular imports
        try:
            from ..utils.config import config_manager
        except ImportError:
            # Handle direct imports when run as a script
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from utils.config import config_manager
        
        # Get API key from various sources
        self.api_key = api_key or config_manager.get_materials_project_api_key()
        
        if not self.api_key:
            logger.warning("No Materials Project API key provided. Some functionality may be limited.")
            logger.warning("Get an API key at https://materialsproject.org/dashboard")
            logger.warning("Set the API key using the config system or MP_API_KEY environment variable")
        
        # Get data directory from config if not provided
        if data_dir is None:
            self.data_dir = os.path.join(
                config_manager.get_config("DEFAULT", "data_dir", 
                                        os.path.expanduser("~/scientific_viz_ai_data")),
                "materials"
            )
        else:
            self.data_dir = data_dir
            
        self.cache = cache
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Base URL for the Materials Project API
        self.base_url = "https://materialsproject.org/rest/v2"
        
        # Cache for data
        self._materials_cache = {}
    
    def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make a request to the Materials Project API.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            
        Returns:
            Response data
        """
        url = f"{self.base_url}/{endpoint}"
        
        # Add API key to payload
        payload["API_KEY"] = self.api_key
        
        # Make request
        try:
            response = requests.post(url, data=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to Materials Project API: {e}")
            return {"error": str(e)}
    
    def get_materials_by_formula(
        self,
        formula: str,
        properties: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get materials data by chemical formula.
        
        Args:
            formula: Chemical formula
            properties: List of properties to retrieve
            
        Returns:
            List of materials data
        """
        # Default properties if none provided
        if properties is None:
            properties = [
                "material_id",
                "pretty_formula",
                "spacegroup",
                "cif",
                "structure",
                "band_gap",
                "density",
                "formation_energy_per_atom",
                "e_above_hull",
                "elasticity",
                "is_stable"
            ]
        
        # Check cache
        cache_key = f"{formula}_{'-'.join(sorted(properties))}"
        if self.cache and cache_key in self._materials_cache:
            return self._materials_cache[cache_key]
        
        # Check if data is already downloaded
        cache_path = os.path.join(self.data_dir, f"{formula.replace('*', 'X')}.json")
        if self.cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                self._materials_cache[cache_key] = data
                return data
            except json.JSONDecodeError:
                logger.warning(f"Error loading cached data from {cache_path}. Downloading fresh data.")
        
        # Prepare request
        payload = {
            "criteria": formula,
            "properties": properties
        }
        
        # Make request
        response = self._make_request("materials/query", payload)
        
        # Process response
        if "response" in response:
            materials = response["response"]
            
            # Cache data
            if self.cache:
                self._materials_cache[cache_key] = materials
                with open(cache_path, "w") as f:
                    json.dump(materials, f)
            
            return materials
        else:
            logger.error(f"Error retrieving materials data: {response.get('error', 'Unknown error')}")
            return []
    
    def get_materials_by_elements(
        self,
        elements: List[str],
        properties: Optional[List[str]] = None,
        max_elements: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get materials data by constituent elements.
        
        Args:
            elements: List of elements
            properties: List of properties to retrieve
            max_elements: Maximum number of elements in the material
            
        Returns:
            List of materials data
        """
        # Default properties if none provided
        if properties is None:
            properties = [
                "material_id",
                "pretty_formula",
                "spacegroup",
                "cif",
                "structure",
                "band_gap",
                "density",
                "formation_energy_per_atom",
                "e_above_hull",
                "elasticity",
                "is_stable"
            ]
        
        # Construct formula query
        formula = {"$and": []}
        for element in elements:
            formula["$and"].append({"elements": {"$eq": element}})
        formula["$and"].append({"nelements": {"$lte": max_elements}})
        
        # Check cache
        elements_key = "-".join(sorted(elements))
        cache_key = f"{elements_key}_{max_elements}_{'-'.join(sorted(properties))}"
        if self.cache and cache_key in self._materials_cache:
            return self._materials_cache[cache_key]
        
        # Check if data is already downloaded
        cache_path = os.path.join(self.data_dir, f"{elements_key}_max{max_elements}.json")
        if self.cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                self._materials_cache[cache_key] = data
                return data
            except json.JSONDecodeError:
                logger.warning(f"Error loading cached data from {cache_path}. Downloading fresh data.")
        
        # Prepare request
        payload = {
            "criteria": json.dumps(formula),
            "properties": properties
        }
        
        # Make request
        response = self._make_request("materials/query", payload)
        
        # Process response
        if "response" in response:
            materials = response["response"]
            
            # Cache data
            if self.cache:
                self._materials_cache[cache_key] = materials
                with open(cache_path, "w") as f:
                    json.dump(materials, f)
            
            return materials
        else:
            logger.error(f"Error retrieving materials data: {response.get('error', 'Unknown error')}")
            return []
    
    def get_xrd_data(
        self,
        material_id: str,
        radiation: str = "CuKa"
    ) -> Dict[str, Any]:
        """
        Get X-ray diffraction data for a material.
        
        Args:
            material_id: Materials Project ID
            radiation: Radiation type (CuKa, MoKa, etc.)
            
        Returns:
            XRD data
        """
        # Check cache
        cache_key = f"xrd_{material_id}_{radiation}"
        if self.cache and cache_key in self._materials_cache:
            return self._materials_cache[cache_key]
        
        # Check if data is already downloaded
        cache_path = os.path.join(self.data_dir, f"xrd_{material_id}_{radiation}.json")
        if self.cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                self._materials_cache[cache_key] = data
                return data
            except json.JSONDecodeError:
                logger.warning(f"Error loading cached data from {cache_path}. Downloading fresh data.")
        
        # Prepare request
        payload = {
            "material_id": material_id,
            "radiation": radiation
        }
        
        # Make request
        response = self._make_request("materials/xrd", payload)
        
        # Process response
        if "response" in response:
            xrd_data = response["response"]
            
            # Cache data
            if self.cache:
                self._materials_cache[cache_key] = xrd_data
                with open(cache_path, "w") as f:
                    json.dump(xrd_data, f)
            
            return xrd_data
        else:
            logger.error(f"Error retrieving XRD data: {response.get('error', 'Unknown error')}")
            return {}
    
    def get_dos_data(
        self,
        material_id: str
    ) -> Dict[str, Any]:
        """
        Get density of states data for a material.
        
        Args:
            material_id: Materials Project ID
            
        Returns:
            DOS data
        """
        # Check cache
        cache_key = f"dos_{material_id}"
        if self.cache and cache_key in self._materials_cache:
            return self._materials_cache[cache_key]
        
        # Check if data is already downloaded
        cache_path = os.path.join(self.data_dir, f"dos_{material_id}.json")
        if self.cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                self._materials_cache[cache_key] = data
                return data
            except json.JSONDecodeError:
                logger.warning(f"Error loading cached data from {cache_path}. Downloading fresh data.")
        
        # Prepare request
        payload = {
            "material_id": material_id
        }
        
        # Make request
        response = self._make_request("materials/dos", payload)
        
        # Process response
        if "response" in response:
            dos_data = response["response"]
            
            # Cache data
            if self.cache:
                self._materials_cache[cache_key] = dos_data
                with open(cache_path, "w") as f:
                    json.dump(dos_data, f)
            
            return dos_data
        else:
            logger.error(f"Error retrieving DOS data: {response.get('error', 'Unknown error')}")
            return {}
    
    def prepare_visualization_data(
        self,
        materials: List[Dict[str, Any]],
        include_xrd: bool = True,
        include_dos: bool = False
    ) -> Dict[str, Any]:
        """
        Prepare data for visualization and analysis.
        
        Args:
            materials: List of materials data
            include_xrd: Whether to include XRD data
            include_dos: Whether to include DOS data
            
        Returns:
            Processed data ready for visualization
        """
        # Filter out materials with missing data
        filtered_materials = []
        for material in materials:
            if all(k in material for k in ["material_id", "pretty_formula", "band_gap", "structure"]):
                filtered_materials.append(material)
        
        logger.info(f"Processing {len(filtered_materials)} materials with complete data")
        
        # Extract basic properties
        property_data = []
        for material in filtered_materials:
            props = {
                "material_id": material["material_id"],
                "formula": material["pretty_formula"],
                "band_gap": material.get("band_gap", 0),
                "density": material.get("density", 0),
                "formation_energy": material.get("formation_energy_per_atom", 0),
                "stability": material.get("e_above_hull", 0),
                "is_stable": material.get("is_stable", False)
            }
            
            # Add elasticity data if available
            if "elasticity" in material and material["elasticity"]:
                props["bulk_modulus"] = material["elasticity"].get("k_vrh", 0)
                props["shear_modulus"] = material["elasticity"].get("g_vrh", 0)
                
            property_data.append(props)
        
        # Convert to DataFrame
        df = pd.DataFrame(property_data)
        
        # Prepare structure data
        structure_data = []
        for material in filtered_materials:
            if "structure" in material:
                struct = material["structure"]
                
                # Extract lattice parameters
                lattice = struct.get("lattice", {})
                
                # Extract atoms and positions
                sites = struct.get("sites", [])
                atom_types = []
                positions = []
                
                for site in sites:
                    element = site.get("label", "")
                    coords = site.get("xyz", [0, 0, 0])
                    
                    # Map element name to atomic number (simplified)
                    atomic_numbers = {
                        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
                        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
                        "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
                        "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40
                    }
                    
                    atom_type = atomic_numbers.get(element, 0)
                    atom_types.append(atom_type)
                    positions.append(coords)
                
                structure_data.append({
                    "material_id": material["material_id"],
                    "formula": material["pretty_formula"],
                    "lattice_parameters": [
                        lattice.get("a", 0),
                        lattice.get("b", 0),
                        lattice.get("c", 0),
                        lattice.get("alpha", 90),
                        lattice.get("beta", 90),
                        lattice.get("gamma", 90)
                    ],
                    "atom_types": atom_types,
                    "positions": positions
                })
        
        # Prepare XRD data if requested
        xrd_data = []
        if include_xrd:
            for material in tqdm(filtered_materials[:20], desc="Fetching XRD data"):  # Limit to first 20 to avoid rate limiting
                material_id = material["material_id"]
                try:
                    xrd = self.get_xrd_data(material_id)
                    
                    if xrd and "pattern" in xrd:
                        # Extract 2theta and intensity
                        two_theta = xrd["pattern"][0]
                        intensity = xrd["pattern"][1]
                        
                        xrd_data.append({
                            "material_id": material_id,
                            "formula": material["pretty_formula"],
                            "two_theta": two_theta,
                            "intensity": intensity
                        })
                    
                    # Sleep to avoid rate limiting
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Error retrieving XRD data for {material_id}: {e}")
        
        # Prepare DOS data if requested
        dos_data = []
        if include_dos:
            for material in tqdm(filtered_materials[:10], desc="Fetching DOS data"):  # Limit to first 10 to avoid rate limiting
                material_id = material["material_id"]
                try:
                    dos = self.get_dos_data(material_id)
                    
                    if dos and "densities" in dos:
                        dos_data.append({
                            "material_id": material_id,
                            "formula": material["pretty_formula"],
                            "energies": dos.get("energies", []),
                            "densities": dos.get("densities", {})
                        })
                    
                    # Sleep to avoid rate limiting
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error retrieving DOS data for {material_id}: {e}")
        
        # Return processed data
        result = {
            "properties": df,
            "structures": structure_data
        }
        
        if xrd_data:
            result["xrd"] = xrd_data
            
        if dos_data:
            result["dos"] = dos_data
            
        return result


def download_example_dataset(
    api_key: Optional[str] = None,
    output_dir: Optional[str] = None,
    elements: Optional[List[str]] = None,
    max_elements: Optional[int] = None,
    include_xrd: bool = True,
    include_dos: bool = False
) -> Dict[str, Any]:
    """
    Download an example dataset for demonstration.
    
    Args:
        api_key: Materials Project API key (if None, will use config or env vars)
        output_dir: Directory to store downloaded data (if None, will use config)
        elements: List of elements to include (if None, will use config)
        max_elements: Maximum number of elements in compounds (if None, will use config)
        include_xrd: Whether to include XRD data
        include_dos: Whether to include DOS data
        
    Returns:
        Processed dataset
    """
    # Import config system
    try:
        from ..utils.config import config_manager
    except ImportError:
        # Handle direct imports when run as a script
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.config import config_manager
    
    # Use config values if parameters not provided
    if elements is None:
        elements_str = config_manager.get_config("materials", "default_elements", "Si,O")
        elements = elements_str.split(",")
    
    if max_elements is None:
        max_elements = int(config_manager.get_config("materials", "max_structures", "2"))
    
    # Create dataset object
    mp_dataset = MaterialsProjectDataset(api_key=api_key, data_dir=output_dir)
    
    # Fetch materials data
    logger.info(f"Fetching materials with elements {elements} (max {max_elements} elements)")
    materials = mp_dataset.get_materials_by_elements(elements, max_elements=max_elements)
    
    if not materials:
        logger.warning(f"No materials found with elements {elements}")
        return {}
    
    logger.info(f"Retrieved {len(materials)} materials containing elements {elements}")
    
    # Prepare visualization data
    dataset = mp_dataset.prepare_visualization_data(
        materials=materials,
        include_xrd=include_xrd,
        include_dos=include_dos
    )
    
    # Save processed dataset
    elements_key = "-".join(sorted(elements))
    output_path = os.path.join(output_dir, f"{elements_key}_dataset.json")
    
    # Converting numpy arrays to lists for JSON serialization
    processed_dataset = {}
    
    # Convert properties DataFrame to dict
    processed_dataset["properties"] = json.loads(dataset["properties"].to_json(orient="records"))
    
    # Process structures
    processed_dataset["structures"] = []
    for struct in dataset["structures"]:
        processed_struct = {
            "material_id": struct["material_id"],
            "formula": struct["formula"],
            "lattice_parameters": struct["lattice_parameters"],
            "atom_types": struct["atom_types"],
            "positions": [list(map(float, pos)) for pos in struct["positions"]]
        }
        processed_dataset["structures"].append(processed_struct)
    
    # Process XRD data if present
    if "xrd" in dataset:
        processed_dataset["xrd"] = []
        for xrd in dataset["xrd"]:
            processed_xrd = {
                "material_id": xrd["material_id"],
                "formula": xrd["formula"],
                "two_theta": list(map(float, xrd["two_theta"])),
                "intensity": list(map(float, xrd["intensity"]))
            }
            processed_dataset["xrd"].append(processed_xrd)
    
    # Process DOS data if present
    if "dos" in dataset:
        processed_dataset["dos"] = []
        for dos in dataset["dos"]:
            processed_dos = {
                "material_id": dos["material_id"],
                "formula": dos["formula"],
                "energies": list(map(float, dos["energies"]))
            }
            
            # Process densities
            processed_densities = {}
            for orbital, values in dos["densities"].items():
                processed_densities[orbital] = list(map(float, values))
            processed_dos["densities"] = processed_densities
            
            processed_dataset["dos"].append(processed_dos)
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(processed_dataset, f, indent=2)
    
    logger.info(f"Saved processed dataset to {output_path}")
    
    return dataset


def get_materials_data(
    max_samples: int = 1000,
    properties: Optional[List[str]] = None,
    elements: Optional[List[str]] = None,
    max_elements: int = 4,
    api_key: Optional[str] = None,
    data_dir: Optional[str] = None,
    cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Get materials data for use in examples and experiments.
    
    Args:
        max_samples: Maximum number of samples to return
        properties: List of properties to retrieve
        elements: List of elements to include
        max_elements: Maximum number of elements in compounds
        api_key: Materials Project API key
        data_dir: Directory to store data
        cache: Whether to use cached data
        
    Returns:
        List of materials data
    """
    # Default properties if not specified
    if properties is None:
        properties = ["material_id", "pretty_formula", "elements", "band_gap", 
                     "formation_energy_per_atom", "e_above_hull", "is_stable"]
    
    # Default elements if not specified
    if elements is None:
        elements = ["Si", "O", "Al", "Fe", "Ca", "Na", "K", "Mg"]
    
    # Create dataset object
    mp_dataset = MaterialsProjectDataset(api_key=api_key, data_dir=data_dir, cache=cache)
    
    # Check if we have an API key
    if not mp_dataset.api_key:
        logger.warning("No Materials Project API key available. Returning synthetic data.")
        # Generate synthetic data
        materials = []
        elements_list = ["Li", "Na", "K", "Mg", "Ca", "Al", "Si", "Fe", "Co", "Ni", "Cu", "Zn"]
        
        for i in range(min(max_samples, 500)):
            # Choose a random number of elements for this compound
            num_elements = np.random.randint(1, min(max_elements, 4) + 1)
            compound_elements = np.random.choice(elements_list, size=num_elements, replace=False)
            
            # Generate random composition
            fractions = np.random.random(num_elements)
            fractions = fractions / fractions.sum()  # Normalize
            
            # Create composition dictionary
            composition = {}
            for j, element in enumerate(compound_elements):
                composition[element] = float(fractions[j])
            
            # Generate random property values
            band_gap = np.random.uniform(0, 6)  # eV
            formation_energy = np.random.uniform(-5, 2)  # eV/atom
            e_above_hull = np.random.exponential(0.1)  # eV/atom
            is_stable = e_above_hull < 0.05
            
            # Create material entry
            material = {
                "material_id": f"mp-{i+1000}",
                "pretty_formula": "".join([f"{el}{int(frac*10) if frac*10 > 1 else ''}" 
                                        for el, frac in composition.items()]),
                "elements": list(compound_elements),
                "composition": composition,
                "band_gap": float(band_gap),
                "formation_energy_per_atom": float(formation_energy),
                "e_above_hull": float(e_above_hull),
                "is_stable": bool(is_stable)
            }
            
            materials.append(material)
        
        logger.info(f"Generated {len(materials)} synthetic materials")
        return materials
    
    # Fetch real data from Materials Project
    try:
        materials = mp_dataset.get_materials_by_elements(elements, properties=properties, max_elements=max_elements)
        
        # Process materials data
        processed_materials = []
        for material in materials[:max_samples]:
            # Extract element composition
            elements_dict = {}
            if "full_formula" in material:
                # Parse formula like "Li4 Fe4 P4 O16" into elements dict
                formula = material["full_formula"]
                parts = formula.split()
                for part in parts:
                    if not part:
                        continue
                    # Extract element and count
                    element = "".join([c for c in part if c.isalpha()])
                    count_str = "".join([c for c in part if c.isdigit()])
                    count = int(count_str) if count_str else 1
                    elements_dict[element] = count
            
            # Calculate total atoms
            total_atoms = sum(elements_dict.values()) if elements_dict else 1
            
            # Convert counts to fractions
            composition = {el: count/total_atoms for el, count in elements_dict.items()}
            
            # Add composition to material data
            material["composition"] = composition
            
            processed_materials.append(material)
        
        logger.info(f"Retrieved {len(processed_materials)} materials from Materials Project")
        return processed_materials
        
    except Exception as e:
        logger.error(f"Error retrieving materials data: {e}")
        logger.warning("Returning synthetic data instead.")
        # Call this function recursively with api_key=None to get synthetic data
        return get_materials_data(max_samples=max_samples, properties=properties, 
                                 elements=elements, max_elements=max_elements, 
                                 api_key=None, data_dir=data_dir, cache=cache)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and process materials data")
    parser.add_argument("--api-key", type=str, help="Materials Project API key")
    parser.add_argument("--output-dir", type=str, default="./data/datasets/materials_project",
                       help="Directory to store data")
    parser.add_argument("--elements", type=str, nargs="+", default=["Si", "O"],
                       help="Elements to include")
    parser.add_argument("--max-elements", type=int, default=2,
                       help="Maximum number of elements")
    parser.add_argument("--xrd", action="store_true", default=True,
                       help="Include XRD data")
    parser.add_argument("--dos", action="store_true", default=False,
                       help="Include DOS data")
    
    args = parser.parse_args()
    
    # Download dataset
    dataset = download_example_dataset(
        api_key=args.api_key,
        output_dir=args.output_dir,
        elements=args.elements,
        max_elements=args.max_elements,
        include_xrd=args.xrd,
        include_dos=args.dos
    )