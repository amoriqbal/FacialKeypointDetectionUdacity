import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import equalize
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2


from torch.utils.data import Dataset, DataLoader

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Normalize image to [0, 1] range
        # matplotlib.imread() returns float in [0, 1], but ensure it explicitly
        # If values are larger than 1, they're likely in [0, 255] so divide by 255
        if image_copy.max() > 1.0:
            image_copy = image_copy / 255.0
        
        # scale keypoints to be centered around 0 with proper normalization
        # Actual statistics from data: mean=49.07, std=13.00
        # Formula: (keypoints - mean) / std
        key_pts_copy = (key_pts_copy - 49.07) / 13.00


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}


class Rotate3D(object):
    """Apply 3D rotation to face using barycentric coordinate-based mesh warping.
    
    ALGORITHM:
    Step 1: Connect keypoints to form triangles (Delaunay, keypoints only)
    Step 2: Add depth to all pixels in triangles based on face structure
    Step 3: Rotate triangles in 3D space using rotation matrices
    Step 4: Project rotated vertices back to 2D
    Step 5: Warp image using barycentric coordinates within triangles
    
    This creates a geometric mesh that rotates naturally in 3D space,
    producing visible face warping that matches the 3D transformation.
    Visualizations are stored for debugging at each step.
    """
    
    def __init__(self, max_angle_x=15, max_angle_y=15, max_angle_z=10, apply_probability=1.0, 
                 min_angle_x=-15, min_angle_y=-15, min_angle_z=-10, debug=False, store_intermediate=False):
        """
        Args:
            max_angle_x: Maximum rotation angle around X-axis (pitch) in degrees
            max_angle_y: Maximum rotation angle around Y-axis (yaw) in degrees  
            max_angle_z: Maximum rotation angle around Z-axis (roll) in degrees
            apply_probability: Probability of applying rotation (0.0 to 1.0). Default 1.0 = always apply
            debug: If True, print detailed logs for debugging
            store_intermediate: If True, store intermediate visualization data
        """
        self.max_angle_x = max_angle_x
        self.max_angle_y = max_angle_y
        self.max_angle_z = max_angle_z
        self.min_angle_x = min_angle_x
        self.min_angle_y = min_angle_y
        self.min_angle_z = min_angle_z
        self.apply_probability = apply_probability
        self.debug = debug
        self.store_intermediate = store_intermediate
        
        # Storage for intermediate results (for visualization)
        self.step1_triangles = None
        self.step2_depth_map = None
        self.step4_projected_pts = None
        self.step4_projected_triangles = None
    
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        # Skip rotation with probability (1 - apply_probability)
        if np.random.random() > self.apply_probability:
            return sample
        
        h, w = image.shape[:2]
        
        if self.debug:
            print(f"\n[ROTATE3D] Starting 3D rotation Transform")
            print(f"  Image shape: {image.shape}, Keypoints: {key_pts.shape}")
        
        try:
            # Get random rotation angles
            angle_x = np.random.uniform(self.min_angle_x, self.max_angle_x)
            angle_y = np.random.uniform(self.min_angle_y, self.max_angle_y)
            angle_z = np.random.uniform(self.min_angle_z, self.max_angle_z)
            
            if self.debug:
                print(f"  Rotation angles: X={angle_x:.1f}°, Y={angle_y:.1f}°, Z={angle_z:.1f}°")
            
            # ========== STEP 1: Triangulation from keypoints only ==========
            triangles = self._step1_triangulate(key_pts, h, w)
            if self.debug:
                print(f"  Step 1: Created {len(triangles)} triangles from {len(key_pts)} keypoints")
            
            # ========== STEP 2: Add depth and create depth map ==========
            keypoints_3d, depth_map = self._step2_add_depth(key_pts, triangles, h, w)
            if self.debug:
                print(f"  Step 2: Depth range: {keypoints_3d[:, 2].min():.1f} to {keypoints_3d[:, 2].max():.1f}")
            
            # ========== STEP 3: Rotate in 3D ==========
            rot_matrix = self._get_rotation_matrix(angle_x, angle_y, angle_z)
            keypoints_3d_rotated = keypoints_3d @ rot_matrix.T
            if self.debug:
                print(f"  Step 3: Applied 3D rotation")
            
            # ========== STEP 4: Project back to 2D ==========
            key_pts_projected = self._step4_project_to_2d(keypoints_3d_rotated, h, w)
            if self.debug:
                print(f"  Step 4: Projected back to 2D")
            
            # ========== STEP 5: Warp image using barycentric coordinates ==========
            image_warped = self._step5_warp_by_barycentric(image, key_pts, key_pts_projected, triangles)
            if self.debug:
                diff = np.abs(image.astype(float) - image_warped.astype(float)).max()
                print(f"  Step 5: Image warping complete. Max pixel diff: {diff:.2f}")
            
            return {'image': image_warped.astype(image.dtype), 'keypoints': key_pts_projected}
        
        except Exception as e:
            if self.debug:
                import traceback
                print(f"  ERROR: {str(e)}")
                traceback.print_exc()
            # Return original if anything fails
            return sample
    
    def _step1_triangulate(self, keypoints, h, w):
        """STEP 1: Create Delaunay triangulation using ONLY keypoints as vertices.
        
        No boundary points - the triangulation covers only the keypoint cloud.
        """
        from scipy.spatial import Delaunay
        
        try:
            # Create Delaunay triangulation using ONLY keypoints
            tri = Delaunay(keypoints)
            triangles = tri.simplices.tolist()
            
            if self.debug:
                print(f"    Delaunay created {len(triangles)} triangles")
            
            if self.store_intermediate:
                self.step1_triangles = triangles
            
            return triangles
        except Exception as e:
            if self.debug:
                print(f"    Triangulation failed: {e}")
            return []
    
    def _step2_add_depth(self, keypoints, triangles, h, w):
        """STEP 2: Add depth to keypoints and create a depth map for all pixels.
        
        Depth is RELATIVE to the face surface (in 3D space):
        - Eyes (top center): depth = 0 (recessed baseline - eye sockets)
        - Nose (center): depth = +40 to +60 (protrudes TOWARDS camera)
        - Lips (bottom center): depth = +20 to +30 (protrudes towards camera)
        - Cheeks (sides): depth = -15 to -30 (recessed AWAY from camera)
        - Chin (bottom): depth = -10 to -20 (slightly recessed)
        
        This creates proper 3D face geometry when rotated.
        """
        # Normalize coordinates to [-1, 1]
        kpts_norm = keypoints.copy()
        kpts_norm[:, 0] = (keypoints[:, 0] - w/2) / (w/2)  # -1 (left) to +1 (right)
        kpts_norm[:, 1] = (keypoints[:, 1] - h/2) / (h/2)  # -1 (top) to +1 (bottom)
        
        # Base depth for face surface
        base_depth = 750  # Camera distance in 3D space
        
        # ========== VERTICAL VARIATION (Y axis) ==========
        # Top of face (eyes, negative Y): close to baseline (depth ~ 0)
        # Center (nose, Y ~ 0): protrudes forward (+depth)
        # Bottom (chin/mouth, positive Y): slightly back (-depth) 
        depth_from_y = np.zeros_like(kpts_norm[:, 1])
        
        # Eyes area (y < -0.3): baseline depth = 0
        mask_eyes = kpts_norm[:, 1] < -0.3
        depth_from_y[mask_eyes] = 0
        
        # Nose area (-0.3 to 0.0): positive depth (protrudes)
        mask_nose = (kpts_norm[:, 1] >= -0.3) & (kpts_norm[:, 1] <= 0.0)
        depth_from_y[mask_nose] = 50 * np.cos(np.pi/2 * (kpts_norm[mask_nose, 1] + 0.3) / 0.3)  # Max 50 at center
        
        # Mouth/lips (0.0 to 0.2): still protrudes but less than nose
        mask_lips = (kpts_norm[:, 1] > 0.0) & (kpts_norm[:, 1] <= 0.2)
        depth_from_y[mask_lips] = 30 * np.exp(-(kpts_norm[mask_lips, 1]**2) * 10)
        
        # Chin (y > 0.2): slightly recessed
        mask_chin = kpts_norm[:, 1] > 0.2
        depth_from_y[mask_chin] = -10
        
        # ========== HORIZONTAL VARIATION (X axis) ==========
        # Center nose (x ~ 0): baseline
        # Cheeks/sides (left/right): recessed away
        depth_from_x = -np.abs(kpts_norm[:, 0])**1.5 * 25  # Sides are -25 when |x|=1
        
        # ========== COMBINE DEPTHS ==========
        depth = base_depth + depth_from_y + depth_from_x
        
        # Ensure reasonable depth range for projection
        depth = np.clip(depth, 700, 800)
        
        # Create 3D keypoints: (x_2d, y_2d, z_3d)
        keypoints_3d = np.column_stack([
            keypoints[:, 0],      # X: original 2D position
            keypoints[:, 1],      # Y: original 2D position  
            depth                 # Z: 3D depth in camera space
        ])
        
        # Create depth map: interpolate depth within each triangle
        depth_map = np.zeros((h, w), dtype=np.float32)
        
        for tri_idx in triangles:
            # Get vertices of this triangle
            v0, v1, v2 = keypoints[tri_idx[0]], keypoints[tri_idx[1]], keypoints[tri_idx[2]]
            d0, d1, d2 = depth[tri_idx[0]], depth[tri_idx[1]], depth[tri_idx[2]]
            
            # Compute bounding box
            x_min = int(max(0, np.floor(np.min([v0[0], v1[0], v2[0]]))))
            x_max = int(min(w-1, np.ceil(np.max([v0[0], v1[0], v2[0]]))))
            y_min = int(max(0, np.floor(np.min([v0[1], v1[1], v2[1]]))))
            y_max = int(min(h-1, np.ceil(np.max([v0[1], v1[1], v2[1]]))))
            
            # For each pixel in the bounding box
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    # Check if pixel is inside triangle using barycentric coordinates
                    bary = self._barycentric_coords(np.array([x, y], dtype=np.float32), v0, v1, v2)
                    
                    if bary is not None and np.all(bary >= -0.01) and np.sum(bary) > 0.99:
                        # Interpolate depth
                        interp_depth = bary[0] * d0 + bary[1] * d1 + bary[2] * d2
                        depth_map[y, x] = interp_depth
        
        if self.debug:
            non_zero = depth_map[depth_map > 0].size
            print(f"    Depth map: {non_zero} pixels, range {depth_map[depth_map > 0].min():.1f} to {depth_map.max():.1f}")
            print(f"    Keypoint depth range: {depth.min():.1f} to {depth.max():.1f}")
        
        if self.store_intermediate:
            self.step2_depth_map = depth_map.copy()
        
        return keypoints_3d, depth_map
    
    def _step4_project_to_2d(self, keypoints_3d, h, w, focal_length=1000):
        """STEP 4: Project 3D points back to 2D using perspective projection."""
        x_3d = keypoints_3d[:, 0]
        y_3d = keypoints_3d[:, 1]
        z_3d = keypoints_3d[:, 2]
        
        # Ensure positive depths
        z_3d = np.maximum(z_3d, 1)
        
        # Perspective projection
        x_2d = (focal_length * x_3d / z_3d) + w/2
        y_2d = (focal_length * y_3d / z_3d) + h/2
        
        # Clip to image bounds
        x_2d = np.clip(x_2d, 0, w - 1)
        y_2d = np.clip(y_2d, 0, h - 1)
        
        if self.debug:
            print(f"    Projected points: x_2d in [{x_2d.min():.1f}, {x_2d.max():.1f}], y_2d in [{y_2d.min():.1f}, {y_2d.max():.1f}]")
        
        return np.column_stack([x_2d, y_2d])
    
    def _step5_warp_by_barycentric(self, image, src_pts, dst_pts, triangles):
        """STEP 5: Warp image using barycentric coordinates within triangles.
        
        For each triangle:
        1. For each pixel in the source triangle, compute barycentric coordinates
        2. Use same coordinates to find corresponding pixel in destination triangle
        3. Sample from source image and place in destination
        """
        h, w = image.shape[:2]
        output = image.copy().astype(np.float32)
        coverage_map = np.zeros((h, w), dtype=np.uint8)
        
        warped_pixels = 0
        
        for tri_idx in triangles:
            # Source triangle vertices
            s0, s1, s2 = src_pts[tri_idx[0]], src_pts[tri_idx[1]], src_pts[tri_idx[2]]
            
            # Destination triangle vertices
            d0, d1, d2 = dst_pts[tri_idx[0]], dst_pts[tri_idx[1]], dst_pts[tri_idx[2]]
            
            # Compute bounding box of destination triangle
            x_min = int(max(0, np.floor(np.min([d0[0], d1[0], d2[0]]))))
            x_max = int(min(w-1, np.ceil(np.max([d0[0], d1[0], d2[0]]))))
            y_min = int(max(0, np.floor(np.min([d0[1], d1[1], d2[1]]))))
            y_max = int(min(h-1, np.ceil(np.max([d0[1], d1[1], d2[1]]))))
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
            # For each pixel in destination triangle bounding box
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    pixel = np.array([x, y], dtype=np.float32)
                    
                    # Get barycentric coordinates relative to destination triangle
                    bary = self._barycentric_coords(pixel, d0, d1, d2)
                    
                    if bary is None or np.any(bary < -0.01) or np.sum(bary) < 0.99:
                        continue
                    
                    # Use same barycentric coordinates to find source pixel
                    src_x = bary[0] * s0[0] + bary[1] * s1[0] + bary[2] * s2[0]
                    src_y = bary[0] * s0[1] + bary[1] * s1[1] + bary[2] * s2[1]
                    
                    # Bilinear interpolation from source image
                    src_x = np.clip(src_x, 0, w - 1)
                    src_y = np.clip(src_y, 0, h - 1)
                    
                    # Get source pixel (simple nearest neighbor for now)
                    src_val = image[int(src_y), int(src_x)]
                    
                    # Place in output
                    output[y, x] = src_val
                    coverage_map[y, x] = 1
                    warped_pixels += 1
        
        if self.debug:
            coverage_pct = 100 * coverage_map.sum() / (h * w)
            print(f"    Warped {warped_pixels} pixels ({coverage_pct:.1f}% coverage)")
        
        return np.uint8(np.clip(output, 0, 255))
    
    def _get_rotation_matrix(self, angle_x_deg, angle_y_deg, angle_z_deg):
        """Get 3D rotation matrix using ZYX Euler angles.
        
        Combines rotations around X (pitch), Y (yaw), and Z (roll) axes.
        """
        angle_x = np.radians(angle_x_deg)
        angle_y = np.radians(angle_y_deg)
        angle_z = np.radians(angle_z_deg)
        
        # Rotation around X-axis (pitch - up/down head tilt)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])
        
        # Rotation around Y-axis (yaw - left/right head turn)
        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])
        
        # Rotation around Z-axis (roll - head tilt/rotation)
        Rz = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation in ZYX order
        return Rz @ Ry @ Rx
    
    def _barycentric_coords(self, p, a, b, c):
        """Compute barycentric coordinates of point p relative to triangle abc."""
        v0 = c - a
        v1 = b - a
        v2 = p - a
        
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        
        denom = dot00 * dot11 - dot01 * dot01
        
        if abs(denom) < 1e-10:
            return None
        
        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w = 1.0 - u - v
        
        return np.array([w, v, u])
    


class CropFaceRegion(object):
    """Crop the face region from the background using keypoints as constraints.
    
    Creates a convex hull or bounding region around keypoints and crops
    the image, setting background to black.
    """
    
    def __init__(self, expansion_factor=1.2):
        """
        Args:
            expansion_factor: How much to expand the convex hull region (1.0 = no expansion)
        """
        self.expansion_factor = expansion_factor
    
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        h, w = image.shape[:2]
        image_copy = np.copy(image).astype(float)
        
        try:
            # Create mask for face region using convex hull of keypoints
            mask = self._create_face_mask(key_pts, h, w, self.expansion_factor)
            
            # Apply mask: set background (mask=0) to black
            image_copy[mask == 0] = 0
            
        except Exception as e:
            # Continue with original image if cropping fails
            pass
        
        return {'image': image_copy.astype(image.dtype), 'keypoints': key_pts}
    
    def _create_face_mask(self, keypoints, h, w, expansion_factor):
        """Create binary mask of face region using keypoints"""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get convex hull of keypoints
        try:
            keypoints_int = np.round(keypoints).astype(np.int32)
            hull = cv2.convexHull(keypoints_int)
            
            # Expand hull slightly
            if expansion_factor > 1.0:
                center = hull.mean(axis=0)
                hull_expanded = (hull - center) * expansion_factor + center
                hull_expanded = np.round(hull_expanded).astype(np.int32)
            else:
                hull_expanded = hull
            
            # Draw filled polygon
            cv2.drawContours(mask, [hull_expanded], 0, 1, thickness=cv2.FILLED)
            
            # Apply morphological operations to smooth edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
        except Exception as e:
            # If hull fails, create circular mask around keypoint centroid
            center = keypoints.mean(axis=0).astype(np.int32)
            radius = int(np.linalg.norm(keypoints - keypoints.mean(axis=0)).mean() * expansion_factor)
            cv2.circle(mask, tuple(center), radius, 1, thickness=cv2.FILLED)
        
        return mask