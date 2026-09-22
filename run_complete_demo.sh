#!/bin/bash
# Complete LIC UV Visualization Demo
# This script runs the complete pipeline to demonstrate LIC for UV visualization

set -e  # Exit on error

echo "========================================="
echo "LIC UV Visualization - Complete Demo"
echo "========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

echo "Step 1: Generating test meshes..."
echo "---------------------------------"
python generate_test_mesh.py demo_cylinder.obj --type cylinder --segments 48
python generate_test_mesh.py demo_sphere.obj --type sphere --segments 48
echo "✓ Generated demo_cylinder.obj and demo_sphere.obj"
echo ""

echo "Step 2: Generating LIC textures for cylinder..."
echo "------------------------------------------------"
echo "  → Cylinder U direction (horizontal flow)"
python lic_uv_demo.py demo_cylinder.obj cylinder_u_lic.png \
    --direction u --resolution 256 --kernel-length 25

echo "  → Cylinder V direction (vertical flow)"
python lic_uv_demo.py demo_cylinder.obj cylinder_v_lic.png \
    --direction v --resolution 256 --kernel-length 25

echo "  → Cylinder both directions"
python lic_uv_demo.py demo_cylinder.obj cylinder_both_lic.png \
    --direction both --resolution 256 --kernel-length 25
echo ""

echo "Step 3: Generating LIC textures for sphere..."
echo "----------------------------------------------"
echo "  → Sphere U direction (longitudinal flow)"
python lic_uv_demo.py demo_sphere.obj sphere_u_lic.png \
    --direction u --resolution 256 --kernel-length 25

echo "  → Sphere V direction (latitudinal flow)"
python lic_uv_demo.py demo_sphere.obj sphere_v_lic.png \
    --direction v --resolution 256 --kernel-length 25
echo ""

echo "========================================="
echo "Demo Complete!"
echo "========================================="
echo ""
echo "Generated files:"
echo "  Meshes:"
echo "    - demo_cylinder.obj"
echo "    - demo_sphere.obj"
echo ""
echo "  Cylinder LIC textures:"
echo "    - cylinder_u_lic.png (U direction - horizontal flow)"
echo "    - cylinder_v_lic.png (V direction - vertical flow)"
echo "    - cylinder_both_lic.png (combined)"
echo ""
echo "  Sphere LIC textures:"
echo "    - sphere_u_lic.png (U direction - longitudinal)"
echo "    - sphere_v_lic.png (V direction - latitudinal)"
echo ""
echo "View the results:"
echo "  - Open the PNG files in any image viewer"
echo "  - Or run: python visualize_lic_results.py cylinder_u_lic.png cylinder_v_lic.png"
echo ""
echo "For your own mesh:"
echo "  python lic_uv_demo.py your_mesh.obj output.png --direction u --resolution 512"
echo ""
