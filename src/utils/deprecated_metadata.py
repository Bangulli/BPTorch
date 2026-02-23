## pip install 'bigpicture_metadata_interface @ git+https://github.com/imi-bigpicture/bigpicture_metadata_interface.git'
import xml.etree.ElementTree as ET
from pprint import pprint
import os
from bigpicture_metadata_interface import BPInterface
from bigpicture_metadata_interface.model import Dataset, StainingList, ChemicalStain, ImmunogenicStain, StainingProcedure

SAMPLE_TEMPLATE = {
        "staining": [],
        "species": [],
        "organ": [],
        "diagnosis": [],
        "staining_code": [],
        "species_code": [],
        "organ_code": []
    },

def load_dataset_meta(path):
    return BPInterface.parse_xml_files(path)

def stainlist2list(sl: StainingList):
    codes = []
    meanings = []
    if type(sl) == StainingList:
        for stain in sl.stains:
            if type(stain) == ImmunogenicStain:
                meanings.append(stain.compound)
            elif type(stain) == ChemicalStain:
                codes.append(stain.compound.code)
                meanings.append(stain.compound.meaning)
        return meanings, codes
    elif type(sl) == StainingProcedure:
        try:
            codes.append(sl.procedure.code)
            meanings.append(sl.procedure.meaning)
        except:
            try:
                meanings.append(sl.procedure)
            except:
                print(sl)
                raise
        return meanings, codes

def get_meta_for_image_ID(ds: Dataset, id: str):
    """Spaghetti map from image id to required meta. Very unstable because the BP datasets are apparently not homogenously populated. My more spaghetti version is much more stable than this.

    Args:
        ds (Dataset): BigPicture Interface object
        id (str): The id of the requested image

    Returns:
        dict: The metadata for the current image.
    """
    ## setup vars
    identified = False
    bb_id = 'unknown'
    species = 'unknown'
    site = 'unknown'
    specimen_id = 'unknown'
    diagnosis = 'unknown'
    sitecodes = 'unknown'

    # find meta
    for img in ds.images.values():
        if not img.identifier==id:continue
        slide_id = img.slide.identifier
        stains = img.slide.staining_information
        for bb in ds.biological_beings.values():
            for curid, specimen in bb.specimens.items():
                for block in list(specimen.blocks.keys()):
                    for slide in specimen.blocks[block].slides:
                        if slide_id == slide:
                            bb_id = bb.identifier
                            species = bb.animal_species
                            try:
                                site = specimen.anatomical_site.meaning
                                sitecodes = specimen.anatomical_site.code
                            except: 
                                site = [s.meaning for s in specimen.anatomical_sites]
                                sitecodes = [s.code for s in specimen.anatomical_sites]
                            specimen_id = curid
                            identified=True
                            break
            if identified: break
        break

    # Find observation
    for obs in ds.observations.values():
        try: ref = list(obs.item.specimens.keys())
        except: ref = obs.item.identifier
        if not (specimen_id in ref or specimen_id==ref):continue
        try: diagnosis = obs.statement.code_attributes['Diagnosis']
        except: diagnosis = [diag for k, diag in obs.statement.custom_attributes.items() if 'diagnosis' in k.lower()]
        break
    
    # parse to dict 
    meta = {
        "species": [species.meaning],
        "organ": [site] if type(site) is str else site,
        "species_code": [species.code],
        "organ_code": [sitecodes] if type(site) is str else site
    }
    meta['staining'], meta['staining_code'] = stainlist2list(stains)
    try: meta['diagnosis'] = [diagnosis.meaning]
    except: meta['diagnosis'] = diagnosis
    return meta

def is_parsable(path):
    try:
        ds = load_dataset_meta(path)
        for file in os.listdir(f"{path}/IMAGES"):
            get_meta_for_image_ID(ds, file)
        return True, None
    except Exception as e: return False, e

if __name__ == '__main__':
    ds = load_dataset_meta('/mnt/nas6/data/BigPicture_CBIR/datasets/aa-Dataset-erd7xy-9ry5yb')
    print(get_meta_for_image_ID(ds, 'IMAGE_2A2QTUJAD'))
