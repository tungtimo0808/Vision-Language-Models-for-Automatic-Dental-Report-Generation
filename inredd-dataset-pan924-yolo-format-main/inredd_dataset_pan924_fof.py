"""
Fiftyone app launcher for inredd_dataset_v1
"""

import fiftyone as fo
import fiftyone.types as fot

def create_or_recreate_dataset(name):
    # Checks if a dataset exists, deletes it if it does, and returns a new o
    if name in fo.list_datasets():
        print(f"Dataset '{name}' already exists. Deleting and recreating.")
        fo.delete_dataset(name)
    
    print(f"Creating new dataset: '{name}'")
    return fo.Dataset(name, persistent=True)

if __name__ == "__main__":
    base_data_path = "."
    image_directory = f"{base_data_path}/images"
    
    # Dataset 1: Mouth Polylines
    mouth_dataset_name = "inredd-mouth-polylines"
    mouth_labels_path = f"{base_data_path}/annotations/mouth_and_teeth_labels.json"
    
    # Dataset 2: Teeth Segmentations
    teeth_dataset_name = "inredd-teeth-segmentations"
    teeth_labels_path = f"{base_data_path}/annotations/teeth_fdi_labels.json"
    
    # Create Dataset 1 (Mouth Polylines)
    print("Processing Dataset 1: Mouth Polylines")
    mouth_dataset = create_or_recreate_dataset(mouth_dataset_name)

    # Add the mouth_and_teeth labels 
    mouth_dataset.add_dir(
        dataset_type=fot.COCODetectionDataset,
        labels_path=mouth_labels_path,
        data_path=image_directory,
        use_polylines=True,
        tolerance=2
    )
    mouth_dataset.save()
    print(f"Dataset '{mouth_dataset_name}' created successfully.\n")

    # Create Dataset 2 (Teeth Segmentations) 
    print("Processing Dataset 2: Teeth Segmentations")
    teeth_dataset = create_or_recreate_dataset(teeth_dataset_name)

    # Add the teeth_fdi_labels 
    teeth_dataset.add_dir(
        dataset_type=fot.COCODetectionDataset,
        labels_path=teeth_labels_path,
        data_path=image_directory,
        use_polylines=True,
        tolerance=2
    )
    teeth_dataset.save()
    print(f"Dataset '{teeth_dataset_name}' created successfully.\n")

    # Launch the FiftyOne App to view both datasets
    dataset = None
    datasets=[mouth_dataset_name, teeth_dataset_name]
    for dataset in datasets:
        dataset = fo.load_dataset(dataset)

    if dataset is not None:
        session = fo.launch_app(dataset=dataset)
        print("FiftyOne App is running with both datasets. Press Ctrl+C in your terminal to exit.")
        session.wait(-1)