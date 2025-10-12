import sys
import numpy as np
import pydicom
import torch
from pathlib import Path
from tqdm import tqdm
from monai.data import MetaTensor

def sort_dicom_files(file_list):
    """
    Sort DICOM files by their physical Z position (ImagePositionPatient[2]).
    Essential for determining the correct anatomical stacking order (Head-to-Feet vs Feet-to-Head).
    """
    def get_z_pos(f):
        try:
            # We only need the header tags
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            return float(ds.ImagePositionPatient[2])
        except Exception:
            return 0.0
    return sorted(file_list, key=get_z_pos)

def make_affine_matrix(ds, z_spacing):
    """
    Construct a 4x4 affine matrix from DICOM metadata.
    Follows standard DICOM -> Patient coordinate system mapping.
    Converts from DICOM LPS to MONAI/Nibabel RAS (flips X and Y).
    """
    # Orientation (Cosine Direction Cosines)
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    rx, ry, rz = iop[0], iop[1], iop[2] # Row Vector (Horizontal)
    cx, cy, cz = iop[3], iop[4], iop[5] # Col Vector (Vertical)
    
    # Spacing
    row_spacing, col_spacing = ds.PixelSpacing
    sz = z_spacing
    
    # Origin
    ox, oy, oz = ds.ImagePositionPatient
    
    # Slice normal vector
    r = np.array([rx, ry, rz])
    c = np.array([cx, cy, cz])
    s = np.cross(r, c)
    
    # Build 4x4 Matrix
    mat = np.eye(4)
    
    # Column 0: Index 0 (W / horizontal step)
    # Convert L to R (flip sign)
    mat[0, 0] = -rx * col_spacing
    mat[1, 0] = -ry * col_spacing
    mat[2, 0] = rz * col_spacing # S is same
    
    # Column 1: Index 1 (H / vertical step)
    # Convert P to A (flip sign)
    mat[0, 1] = -cx * row_spacing
    mat[1, 1] = -cy * row_spacing
    mat[2, 1] = cz * row_spacing # S is same
    
    # Column 2: Index 2 (D / slice step)
    mat[0, 2] = -s[0] * sz
    mat[1, 2] = -s[1] * sz
    mat[2, 2] = s[2] * sz
    
    # Origin: Convert L -> R and P -> A
    mat[0, 3] = -ox
    mat[1, 3] = -oy
    mat[2, 3] = oz
    
    return mat

def load_dicom_volume_robust(files):
    """
    Manually load a list of DICOM files into a 3D MONAI MetaTensor.
    Guarantees correct Z-sorting, Float32 dtype, and valid Affine construction.
    
    Optimized for memory efficiency using pre-allocation and in-place operations.
    """
    if not files:
        raise ValueError("No matching DICOM files found provided.")

    files = sort_dicom_files(files) # Ensure sorted by Z
    
    # Load first slice for metadata
    ds_first = pydicom.dcmread(files[0])
    ds_last = pydicom.dcmread(files[-1])
    
    # Calculate Z-spacing robustly
    z1 = float(ds_first.ImagePositionPatient[2])
    z2 = float(ds_last.ImagePositionPatient[2])
    # Average thickness vs header thickness
    thickness = abs(z2 - z1) / (len(files) - 1) if len(files) > 1 else float(getattr(ds_first, 'SliceThickness', 1.0))
    if thickness == 0: thickness = 1.0 # Fallback safety
    
    # Pre-allocate array: (W, H, D) format to match Column 0=RowVector, Column 1=ColVector
    rows = int(ds_first.Rows)
    cols = int(ds_first.Columns)
    depth = len(files)
    
    # Use Float32 to reduce memory footprint
    vol_np = np.zeros((cols, rows, depth), dtype=np.float32)

    # Use tqdm w/ stdout if many slices
    pbar = tqdm(files, desc="Loading slices", file=sys.stdout, leave=False) if len(files) > 100 else files
    
    for i, f in enumerate(pbar):
        d = pydicom.dcmread(f)
        slope = getattr(d, 'RescaleSlope', 1.0)
        intercept = getattr(d, 'RescaleIntercept', 0.0)
        
        pix = d.pixel_array.astype(np.float32)
        pix = pix * slope + intercept
        
        # Robust dimensionality check
        if pix.shape == (rows, cols):
            # Transpose (H, W) -> (W, H) to match index order 0=W, 1=H
            vol_np[:, :, i] = pix.T
    
    # Handle NaNs in-place to avoid copy
    if np.isnan(vol_np).any():
        print("      Sanitizing volume (removing NaNs)...", end="", flush=True)
        vol_np = np.nan_to_num(vol_np, copy=False)
        print(" Done.")
    
    # Add Channel Dims -> (1, H, W, D)
    vol_np = vol_np[np.newaxis, ...]   
    
    # Construct Affine Matrix
    affine = make_affine_matrix(ds_first, thickness)
    
    # Return MONAI MetaTensor
    return MetaTensor(torch.tensor(vol_np), affine=torch.tensor(affine))


# ==============================================================================
# ROBUST SEGMENTATION LOADING (Added for Mask Alignment Fix)
# ==============================================================================

def load_dicom_seg_reconstructed(seg_path, ct_files, target_shape, ct_affine=None):
    """
    Manually reconstructs a Segmentation Volume from a DICOM SEG file
    by mapping frames to the corresponding CT slices via SOPInstanceUID.
    
    Args:
        seg_path (Path): Path to DICOM SEG file.
        ct_files (list): List of paths to CT DICOM files (must be sorted by Z).
        target_shape (tuple): (1, H, W, D) target shape.
        ct_affine (ndarray, optional): CT Affine for reference (unused except for interface compat).
        
    Returns:
        torch.Tensor: (1, H, W, D) aligned segmentation mask (raw spacing).
    """
    import pydicom
    import numpy as np
    import torch
    
    # 1. Map CT Ids to Index
    sop_to_idx = {}
    for idx, f in enumerate(ct_files):
        try:
            d = pydicom.dcmread(f, stop_before_pixels=True)
            sop_to_idx[d.SOPInstanceUID] = idx
        except Exception:
            pass
            
    # 2. Read SEG
    seg_ds = pydicom.dcmread(seg_path)
    
    rows = seg_ds.Rows
    cols = seg_ds.Columns
    num_frames = getattr(seg_ds, "NumberOfFrames", 1)
    
    # Depth derived from CT
    D_ct = target_shape[-1]
    
    # Alloc reconstruction buffer [W_seg, H_seg, D_ct]
    recon_vol = np.zeros((cols, rows, D_ct), dtype=np.uint8)
    
    # 3. Iterate Frames
    per_frame = getattr(seg_ds, "PerFrameFunctionalGroupsSequence", None)
    pixel_array = seg_ds.pixel_array # (Frames, H, W)
    
    for i in range(num_frames):
        ref_uid = None
        
        # Check PerFrame
        if per_frame and i < len(per_frame):
             grp = per_frame[i]
             if 'DerivationImageSequence' in grp:
                 for deriv in grp.DerivationImageSequence:
                     if 'SourceImageSequence' in deriv:
                         for src in deriv.SourceImageSequence:
                             if 'ReferencedSOPInstanceUID' in src:
                                 ref_uid = src.ReferencedSOPInstanceUID
                                 break
        
        if ref_uid and ref_uid in sop_to_idx:
            z_idx = sop_to_idx[ref_uid]
            # Accumulate mask (Logical OR if overlap)
            # Transpose (H, W) -> (W, H)
            mask_slice = pixel_array[i].T
            if mask_slice.max() > 0:
                 recon_vol[:, :, z_idx] = np.maximum(recon_vol[:, :, z_idx], mask_slice)
            
    # 4. Convert to Tensor
    vol_tensor = torch.tensor(recon_vol.astype(np.float32)).unsqueeze(0) # (1, W_s, H_s, D_ct)
    
    return vol_tensor

def auto_align_orientation(ct_tensor, seg_tensor):
    """
    Heuristic Alignment: Defines the best orientation by maximizing overlap 
    between the Segmentation Mask and Bone-like structures in CT.
    
    Args:
        ct_tensor (Tensor): (1, H, W, D) aligned CT data (MetaTensor or Tensor).
        seg_tensor (Tensor): (1, H, W, D) candidate SEG data (Tensor).
        
    Returns:
        best_seg_tensor (Tensor): Optimally oriented SEG.
    """
    import numpy as np
    import torch
    
    # Bone Threshold (Hounsfield Units). 
    # Assume generic HU range. Bone ~ > 200.
    
    # We work on CPU/Numpy for check
    ct_np = ct_tensor[0].cpu().numpy() if hasattr(ct_tensor, 'cpu') else ct_tensor[0]
    seg_np = seg_tensor[0].cpu().numpy() if hasattr(seg_tensor, 'cpu') else seg_tensor[0]
    
    # Generate Thresholded Bone Mask
    bone_mask = (ct_np > 200).astype(np.float32)
    
    transforms_to_try = [
        ([], "Original"),
        ([0], "Flip Y (Height)"),
        ([1], "Flip X (Width)"),
        ([0, 1], "Flip XY (Both)")
    ]
    
    best_score = -1.0
    best_seg = seg_tensor
    
    for axes, name in transforms_to_try:
        if not axes:
            candidate = seg_np
        else:
            candidate = np.flip(seg_np, axis=axes)
            
        # Compute Overlap Fraction: (Mask & Bone) / Mask
        mask_sum = candidate.sum()
        if mask_sum == 0:
            score = 0
        else:
            overlap = (candidate * bone_mask).sum()
            score = overlap / mask_sum
            
        print(f"      - {name}: Overlap Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_seg = torch.tensor(candidate.copy()).unsqueeze(0)
            
    print(f"      ✅ Selected Orientation Score: {best_score:.4f}")
    return best_seg


# ==============================================================================
# CLASS MAPPING HELPERS (Added for Spine Grouping Fix)
# ==============================================================================

# Mapping from common spine labels to MONAI Whole Body Model indices (from metadata.json)
VERTEBRAE_NAME_TO_INDEX = {
    "C1": 41, "C2": 40, "C3": 39, "C4": 38, "C5": 37, "C6": 36, "C7": 35,
    "T1": 34, "T2": 33, "T3": 32, "T4": 31, "T5": 30, "T6": 29, "T7": 28, "T8": 27, "T9": 26, "T10": 25, "T11": 24, "T12": 23,
    "L1": 22, "L2": 21, "L3": 20, "L4": 19, "L5": 18,
    "SACRUM": 92
}

def get_annotated_spine_indices(seg_path):
    """
    Parses a DICOM SEG file to identify which vertebral levels are annotated.
    Maps labels like 'T1 vertebra' or 'L2' to model indices.
    
    Returns:
        list of int: Indices of the annotated vertebral classes.
    """
    try:
        ds = pydicom.dcmread(seg_path, stop_before_pixels=True)
        if not hasattr(ds, "SegmentSequence"):
            return []
            
        found_indices = []
        for segment in ds.SegmentSequence:
            label = getattr(segment, "SegmentLabel", "").upper()
            # Clean label: "T1 VERTEBRA" -> "T1", "L2 LEVEL" -> "L2"
            # We look for matches in our dictionary
            for key, idx in VERTEBRAE_NAME_TO_INDEX.items():
                # Exact match or word boundary match to avoid T1 matching T10
                import re
                if re.search(r'\b' + key + r'\b', label):
                    found_indices.append(idx)
                    break
        
        # Remove duplicates
        return sorted(list(set(found_indices)))
    except Exception as e:
        print(f"      ⚠️ Warning: Failed to parse SEG metadata: {e}")
        return []

def find_ct_and_seg_files(patient_dir):
    """
    Heuristic to separate CT series from SEG series/files.
    Assumes CT series are in folders NOT named 'SEG'.
    Assumes SEG files might be in folders named 'SEG' or have 'SEG' in filename.
    Returns:
        ct_files (list): List of Path objects for CT.
        seg_candidates (list): List of Path objects for SEG.
    """
    ct_files = []
    seg_candidates = []

    # Recursive search
    all_files = sorted(list(patient_dir.rglob("*.dcm")))
    
    for f in all_files:
        # Check parent folder name and filename for "SEG"
        is_seg = "SEG" in str(f).upper() or "SEG" in f.parent.name.upper()
        if is_seg:
            # Verify it is actually a file
            if f.is_file():
                seg_candidates.append(f)
        else:
            if f.is_file():
                ct_files.append(f)
            
    return ct_files, seg_candidates
