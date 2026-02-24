from bigpicture_metadata_interface.model.common.values import Code
from bigpicture_metadata_interface.model.dataset import Dataset
from bigpicture_metadata_interface.model.image import Image
from bigpicture_metadata_interface.model.observation import Observation
from bigpicture_metadata_interface.model.sample import BiologicalBeing, Specimen
from bigpicture_metadata_interface import BPInterface
from bigpicture_metadata_interface.model.stain import (
    ChemicalStain,
    ImmunogenicStain,
    InSituHybridisationStain,
    Stain,
    StainingList,
    StainingProcedure,
)
import os
from pathlib import Path

# TODO:
#   - Fix diagnosis

class BPMeta():
    """revised version of my deprecated_metadata.py on GitHub: https://github.com/imi-bigpicture/bigpicture-metadata-interface/issues/93#issue-3886463132
    """
    
    @staticmethod
    def get_supported_fields():
        return ['staining', 'diagnosis', 'organ', 'species']
    
    def __init__(self, ds):
        self.ds = self._load_dataset_meta(ds)
        self.mappings = self._create_mappings(self.ds)
        self.lookup = self._make_lookup(ds)
        
    def __getitem__(self, key):
        if key not in list(self.lookup.keys()): self.lookup[key]=self._get_meta_for_image_ID(self.ds, key, *self.mappings)
        return self.lookup[key]
    
    def _make_lookup(self, ds):
        lookup = {}
        for img in os.listdir(Path(ds)/'IMAGES'):
            lookup[img]=self._get_meta_for_image_ID(self.ds, img, *self.mappings)
        return lookup
    
    def get_beings(self):
        beings = []
        images = {}
        for img, v in self.lookup.items():
            if v['being'] not in beings: 
                beings.append(v['being'])
                images[v['being']] = [img]
            else: images[v['being']].append(img)
        return beings, images
    
    def _load_dataset_meta(self, path):
        self.root = path
        return BPInterface.parse_xml_files(path)

    def _create_mappings(
        self,
        ds: Dataset,
    ) -> tuple[dict[str, BiologicalBeing], dict[str, Specimen], dict[str, Observation]]:
        """Create a mapping from slide id to biological being and specimen and from specimen id to observation."""
        biological_being_mapping: dict[str, BiologicalBeing] = {}
        specimen_mapping: dict[str, Specimen] = {}
        observation_mapping: dict[str, Observation] = {}
        for biological_being in ds.biological_beings.values():
            for specimen in biological_being.specimens.values():
                for block in specimen.blocks.values():
                    for slide in block.slides.values():
                        biological_being_mapping[slide.identifier] = biological_being
                        specimen_mapping[slide.identifier] = specimen
        for observation in ds.observations.values():
            if isinstance(observation.item, Specimen):
                observation_mapping[observation.item.identifier] = observation
        return biological_being_mapping, specimen_mapping, observation_mapping
        
    def _get_diagnosis(self, observation: Observation):
        try:
            try: diagnosis = observation.statement.code_attributes['Diagnosis']
            except: diagnosis = [diag for k, diag in observation.statement.custom_attributes.items() if 'diagnosis' in k.lower()]
        except:
            try: diagnosis = [diagnose for tag, diagnose in observation.statement.custom_attributes.items() if "diagnosis" in tag.lower() and isinstance(diagnose, str)]
            except: diagnosis = 'unknown'
        return diagnosis


    def _string_or_code_to_string(self, value: str | Code) -> str:
        if isinstance(value, Code):
            return value.meaning
        return value


    def _get_stain_description(self, stain: Stain):
        if isinstance(stain, ChemicalStain):
            return self._string_or_code_to_string(stain.compound)
        elif isinstance(stain, (ImmunogenicStain, InSituHybridisationStain)):
            return self._string_or_code_to_string(stain.target)
        raise ValueError(f"Unknown stain type: {type(stain)}")


    def _get_staining_description(self, image: Image) -> str | list[str]:
        if isinstance(image.slide.staining_information, StainingList):
            return [
                self._get_stain_description(stain)
                for stain in image.slide.staining_information.stains
            ]

        elif isinstance(image.slide.staining_information, StainingProcedure):
            return self._string_or_code_to_string(image.slide.staining_information.procedure)
        else:
            raise ValueError(
                f"Unknown staining type: {type(image.slide.staining_information)}"
            )


    def _get_meta_for_image_ID(
        self,
        ds: Dataset,
        image_accession: str,
        biological_being_mapping: dict[str, BiologicalBeing],
        specimen_mapping: dict[str, Specimen],
        observation_mapping: dict[str, Observation],
    ) -> dict:
        """Spaghetti map from image id to required meta. Very unstable because the BP datasets are apparently not homogenously populated

        Args:
            ds (Dataset): BigPicture Interface object
            image_accession (str): The id of the requested image
            biological_being_mapping (dict[str, BiologicalBeing]): Mapping from slide id to biological being
            specimen_mapping (dict[str, Specimen]): Mapping from slide id to specimen
            observation_mapping (dict[str, Observation]): Mapping from specimen id to observation

        Returns:
            dict: The metadata for the current image.
        """
        try:
            image = next(
                image for image in ds.images.values() if image.identifier == image_accession
            )
        except StopIteration:
            raise ValueError(f"Image with id {image_accession} not found in dataset.")

        try:
            biological_being = biological_being_mapping[image.slide.identifier]
        except KeyError:
            raise ValueError(
                f"Biological being for slide with id {image.slide.identifier} for image {image_accession} not found in dataset."
            )
        try:
            specimen = specimen_mapping[image.slide.identifier]
        except KeyError:
            raise ValueError(
                f"Specimen for slide with id {image.slide.identifier} for image {image_accession} not found in dataset."
            )
        observation = observation_mapping.get(specimen.identifier, None)
        if observation is not None:
            diagnosis = self._get_diagnosis(observation)
        else:
            diagnosis = "unknown"

        staining_description = self._get_staining_description(image)

        # parse to dict
        meta = {
            "being": biological_being.identifier,
            "species": biological_being.animal_species.meaning,
            "organ": (
                specimen.anatomical_site.meaning if specimen.anatomical_site else "unknown"
            ),
            "species_code": biological_being.animal_species.code,
            "organ_code": (
                specimen.anatomical_site.code if specimen.anatomical_site else "unknown"
            ),
            "diagnosis": diagnosis,
            "staining": staining_description,
        }
        return meta

