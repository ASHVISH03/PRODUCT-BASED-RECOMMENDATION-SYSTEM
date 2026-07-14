"""
Fallback Dataset Generator
===========================
Generates a synthetic product dataset that matches the schema of the
Kaggle Amazon Sales Dataset. This is used ONLY as a development fallback
when Kaggle credentials are not available.

The synthetic data:
- Matches the exact column schema of the real dataset
- Contains products across realistic categories
- Includes realistic price ranges, ratings, and descriptions
- Should NEVER be used for production training or evaluation

Usage:
    python scripts/generate_fallback.py

Output:
    data/external/sample_products.csv

After running, set 'use_fallback: true' in src/config/config.yaml
to enable the fallback in the pipeline.
"""

import csv
import io
import random
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows to prevent UnicodeEncodeError from print()
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_FILE = PROJECT_ROOT / "data" / "external" / "sample_products.csv"

random.seed(42)


# ---------------------------------------------------------------
# Product definitions by category
# ---------------------------------------------------------------

CATEGORIES = {
    "Electronics|Smartphones & Accessories": [
        ("Samsung Galaxy S24 Ultra 5G", "Samsung", 89999, 109999,
         "Flagship smartphone with 200MP camera, S Pen, 5000mAh battery, 12GB RAM, 256GB storage."),
        ("Apple iPhone 15 Pro", "Apple", 134900, 149900,
         "Titanium design, A17 Pro chip, ProMotion display, 48MP main camera, USB-C."),
        ("OnePlus 12 5G", "OnePlus", 64999, 74999,
         "Snapdragon 8 Gen 3, 50MP Hasselblad camera, 100W SUPERVOOC charging, 5400mAh."),
        ("Xiaomi 14", "Xiaomi", 59999, 69999,
         "Leica optics, Snapdragon 8 Gen 3, 120W HyperCharge, 50MP triple camera."),
        ("Google Pixel 8 Pro", "Google", 84999, 99999,
         "Google Tensor G3 chip, 50MP camera, 7 years updates, temperature sensor."),
        ("Realme GT 5 Pro", "Realme", 39999, 49999,
         "Snapdragon 8 Gen 3, 50MP camera, 100W fast charging, 5000mAh battery."),
        ("Motorola Edge 50 Pro", "Motorola", 31999, 39999,
         "144Hz pOLED display, 50MP Sony sensor, 125W TurboPower charging."),
        ("Vivo V30 Pro 5G", "Vivo", 35999, 44999,
         "50MP ZEISS portrait camera, Dimensity 8200, curved AMOLED, 80W charging."),
        ("Nothing Phone 2a", "Nothing", 24999, 29999,
         "Dimensity 7200 Pro, Glyph Interface, 50MP camera, 5000mAh, 45W charging."),
        ("iQOO 12 5G", "iQOO", 52999, 64999,
         "Snapdragon 8 Gen 3, 50MP Zeiss camera, 120W FlashCharge, 5000mAh."),
        ("Poco X6 Pro 5G", "Poco", 26999, 34999,
         "Dimensity 8300 Ultra, 64MP OIS camera, 67W SUPERCHARGE, 5000mAh AMOLED."),
        ("Oppo Reno 11 Pro", "Oppo", 39999, 49999,
         "50MP Triple camera, Hasselblad colour tuning, 80W SUPERVOOC, 4600mAh."),
    ],
    "Electronics|Laptops & Accessories": [
        ("Dell XPS 15 9530", "Dell", 179990, 209990,
         "Intel Core i7-13700H, NVIDIA RTX 4060, 15.6 inch OLED 3.5K display, 16GB DDR5, 512GB SSD."),
        ("Apple MacBook Pro 14 M3", "Apple", 199900, 229900,
         "Apple M3 chip, 14.2 inch Liquid Retina XDR, 18GB unified memory, 1TB SSD, 22hr battery."),
        ("HP Spectre x360 14", "HP", 149999, 179999,
         "Intel EVO i7-1355U, 2.8K OLED touch display, 16GB, 1TB SSD, 360 degree hinge."),
        ("Lenovo ThinkPad X1 Carbon Gen 11", "Lenovo", 159999, 194999,
         "Intel Core i7-1365U, 14 inch 2.8K OLED, 16GB LPDDR5, 512GB SSD, military grade."),
        ("ASUS ROG Zephyrus G14", "ASUS", 129999, 154999,
         "AMD Ryzen 9 7940HS, RTX 4060, 14 inch QHD+ 165Hz, 16GB DDR5, 1TB SSD."),
        ("Acer Swift X 14", "Acer", 79999, 94999,
         "Intel Core i5-13500H, RTX 3050, 14 inch 2.8K OLED, 16GB, 512GB SSD."),
        ("MSI Stealth 15", "MSI", 109999, 134999,
         "Intel Core i7-13700H, RTX 4060, 15.6 inch QHD 165Hz, 16GB DDR5, 1TB NVMe."),
        ("Samsung Galaxy Book3 Pro 360", "Samsung", 149999, 174999,
         "Intel Core i7-1360P, 16 inch 3K AMOLED touch, 16GB, 512GB SSD, S Pen."),
        ("Microsoft Surface Laptop 5", "Microsoft", 124999, 149999,
         "Intel Core i7-1265U, 13.5 inch PixelSense, 16GB, 512GB removable SSD, Windows 11."),
        ("LG Gram 16", "LG", 104999, 124999,
         "Intel Core i7-1360P, 16 inch IPS, 16GB LPDDR5, 1TB SSD, under 1.2kg military grade."),
    ],
    "Electronics|Headphones & Earphones": [
        ("Sony WH-1000XM5", "Sony", 26990, 34990,
         "Industry-leading noise cancellation, 30hr battery, multipoint connection, Hi-Res Audio."),
        ("Apple AirPods Pro 2nd Gen", "Apple", 24900, 29900,
         "Active Noise Cancellation, Transparency mode, Adaptive Audio, H2 chip, MagSafe."),
        ("Bose QuietComfort 45", "Bose", 22900, 29900,
         "World-class noise cancellation, 24hr battery, comfortable design, Aware mode."),
        ("Jabra Evolve2 85", "Jabra", 27999, 34999,
         "ANC, 8-mic array for calls, 37hr battery, multipoint, Teams/Zoom certified."),
        ("Sennheiser Momentum 4 Wireless", "Sennheiser", 24990, 32990,
         "60hr battery, adaptive noise cancellation, foldable design, sound personalization."),
        ("OnePlus Buds Pro 2", "OnePlus", 9999, 12999,
         "47dB ANC, Dynaudio tuning, 12.4mm + 6mm dual drivers, 39hr total battery."),
        ("Samsung Galaxy Buds2 Pro", "Samsung", 11999, 17999,
         "24-bit Hi-Fi audio, ANC, Auto Switch, 360 Audio, IPX7, 8hr battery."),
        ("Boat Rockerz 550", "Boat", 2499, 4999,
         "50mm drivers, 20hr battery, 40dB ANC, foldable design, dual pairing."),
        ("JBL Quantum 350", "JBL", 5999, 8999,
         "48kHz audio, 2.4GHz wireless, JBL QuantumSurround, 22hr battery."),
        ("Realme Buds Air 5", "Realme", 2999, 5499,
         "50dB ANC, 12.4mm bass boost driver, 40hr total battery, IP55 rating."),
    ],
    "Electronics|Televisions": [
        ("Samsung 65 Neo QLED 8K QN800C", "Samsung", 249999, 299999,
         "8K NeoQLED, Neural Quantum Processor 8K, Quantum Matrix Technology Pro, Object Tracking Sound."),
        ("LG OLED65C3PSA 65 inch OLED", "LG", 189999, 219999,
         "OLED evo panel, AI Processor Gen6, Dolby Vision IQ, 4K 120Hz, G-Sync, FreeSync."),
        ("Sony KD-65X90L Bravia 65 inch", "Sony", 149999, 179999,
         "4K Full Array LED, XR Cognitive Processor, Bravia CORE, HDMI 2.1, Google TV."),
        ("OnePlus 65 Y1S Pro", "OnePlus", 54999, 69999,
         "4K UHD, Gamma Color Magic engine, Dolby Audio, 3-side bezel-less design, Android TV."),
        ("Mi Smart TV X65", "Xiaomi", 69999, 84999,
         "65 inch 4K UHD, MEMC, Dolby Vision, 30W Dolby Audio, PatchWall Android TV."),
        ("TCL 65 inch QLED 4K TV", "TCL", 59999, 74999,
         "QLED panel, AiPQ Engine, Dolby Vision, Android TV 11, built-in Chromecast."),
        ("Vu 65 inch Premium 4K Smart TV", "Vu", 49999, 64999,
         "4K HDR, 60W Dolby Atmos sound, 10-core processor, Android TV, 100Hz display."),
    ],
    "Electronics|Cameras & Photography": [
        ("Canon EOS R50 Mirrorless Camera", "Canon", 64999, 79999,
         "24.2MP APS-C sensor, 4K video, Eye AF, dual pixel autofocus, compact design."),
        ("Sony Alpha A7 IV Full Frame", "Sony", 219999, 259999,
         "33MP full-frame BSI CMOS, 4K 60p video, real-time tracking AF, 5-axis stabilisation."),
        ("Nikon Z50 Mirrorless Camera", "Nikon", 74999, 89999,
         "20.9MP DX format, 4K UHD video, 209-point AF, flip touchscreen, dual UHS-I slots."),
        ("Fujifilm X-T5 Mirrorless", "Fujifilm", 159999, 189999,
         "40.2MP X-Trans CMOS 5 HR sensor, 6.2K video, 5-axis IBIS, Film Simulation modes."),
        ("GoPro HERO12 Black Action Cam", "GoPro", 34999, 42999,
         "5.3K60 video, 27MP photos, HyperSmooth 6.0, waterproof to 10m, TimeWarp 3.0."),
    ],
    "Electronics|Tablets": [
        ("Apple iPad Pro 12.9 inch M2", "Apple", 112900, 134900,
         "M2 chip, 12.9 inch Liquid Retina XDR display, ProMotion 120Hz, 5G capable, USB-C."),
        ("Samsung Galaxy Tab S9 Ultra", "Samsung", 108999, 132999,
         "14.6 inch Dynamic AMOLED 2X, Snapdragon 8 Gen 2, S Pen included, IP68."),
        ("OnePlus Pad Go", "OnePlus", 19999, 24999,
         "11.35 inch 2.4K 90Hz display, Helio G99, 8000mAh battery, Dolby Atmos."),
        ("Xiaomi Pad 6", "Xiaomi", 28999, 34999,
         "11 inch 2.8K 144Hz, Snapdragon 870, 8840mAh, 33W charging, 4 speaker array."),
        ("Lenovo Tab P12 Pro", "Lenovo", 59999, 74999,
         "12.6 inch Super AMOLED 2K, Snapdragon 870, 10200mAh, pen and keyboard support."),
    ],
    "Clothing|Men's Clothing": [
        ("Levis 511 Slim Fit Jeans", "Levis", 2999, 4499,
         "Slim fit through seat and thigh, straight leg, stretch fabric, 5-pocket styling."),
        ("Allen Solly Formal Shirt", "Allen Solly", 1299, 2199,
         "Regular fit, full sleeves, solid color, premium cotton blend, office wear."),
        ("HnM Regular Fit Oxford Shirt", "HnM", 1499, 2499,
         "Regular fit, button-down collar, chest pocket, 100% cotton, machine washable."),
        ("US Polo Assn Polo T-shirt", "US Polo Assn", 999, 1799,
         "Classic fit, embroidered logo, premium pique fabric, ribbed collar and cuffs."),
        ("Van Heusen Formal Trousers", "Van Heusen", 1799, 2999,
         "Regular fit, flat front, moisture wicking, 4-way stretch, wrinkle-free finish."),
        ("Peter England Blazer", "Peter England", 4999, 7999,
         "Single-breasted, notch lapel, two-button closure, slim fit, formal occasion."),
        ("Arrow Sports Chino", "Arrow", 1599, 2599,
         "Slim fit chino, stretch cotton, 5-pocket design, versatile casual to smart casual."),
    ],
    "Clothing|Women's Clothing": [
        ("Biba Anarkali Kurta", "Biba", 1499, 2499,
         "Cotton blend, printed design, 3/4 sleeves, round neck, comfortable fit for daily wear."),
        ("W for Woman Flared Kurta", "W", 1299, 2199,
         "Pure cotton, floral print, straight cut with flared bottom, comes with matching dupatta."),
        ("Global Desi Maxi Dress", "Global Desi", 1699, 2799,
         "Georgette fabric, paisley print, flared silhouette, ethnic appeal with modern style."),
        ("Zara Floral Midi Skirt", "Zara", 1999, 3499,
         "Floral print, midi length, elastic waist, fully lined, flowing fit."),
        ("HnM Slim Fit Trousers", "HnM", 1299, 2199,
         "Slim fit, high waist, invisible side pockets, back pockets, stretch blend."),
        ("Fabindia Cotton Salwar Suit Set", "Fabindia", 2499, 3999,
         "Block print, unstitched 3-piece set, kota doria fabric, traditional handcraft."),
    ],
    "Home & Kitchen|Kitchen Appliances": [
        ("Instant Pot Duo 7-in-1", "Instant Pot", 8999, 12999,
         "7-in-1 electric pressure cooker, slow cooker, rice cooker, steamer, saute, yogurt maker."),
        ("Philips Air Fryer HD9200", "Philips", 6999, 9999,
         "Rapid Air technology, 1400W, 4.1L capacity, digital display, 7 presets."),
        ("Inalsa Food Processor Fiesta", "Inalsa", 3999, 5999,
         "600W, 1.5L bowl, 5 accessories including blade, disc, whipper, dough hook."),
        ("Havells Hand Blender Hexo Plus", "Havells", 1799, 2799,
         "700W motor, 3-speed control with turbo, stainless steel shaft, 600ml beaker."),
        ("Prestige IRIS 750W Mixer Grinder", "Prestige", 2999, 4299,
         "750W motor, 3 stainless steel jars, 3 speed with pulse, 5 year warranty."),
        ("Bajaj Majesty 1000 TSS Sandwich Maker", "Bajaj", 999, 1699,
         "1000W, non-stick plates, compact design, indicator lights, 2 year warranty."),
        ("Morphy Richards Fresco 25L Microwave", "Morphy Richards", 8999, 12999,
         "25L solo microwave, 800W, 5 power levels, defrost function, child lock."),
    ],
    "Home & Kitchen|Home Decor": [
        ("Amazon Basics Microfiber Bedsheet Set", "AmazonBasics", 999, 1999,
         "King size, 300 thread count, 4 piece set with pillow covers, machine washable."),
        ("Nilkamal Plastic Folding Table", "Nilkamal", 2499, 3499,
         "Weather-resistant, foldable design, 4 seater, easy to clean, indoor and outdoor use."),
        ("FabIndia Handblock Print Cushion Covers", "FabIndia", 499, 899,
         "Set of 5, 45x45cm, 100% cotton, traditional block print, hand-crafted."),
        ("Bombay Dyeing Towel Set", "Bombay Dyeing", 699, 1299,
         "Set of 4, 500 GSM, quick-dry, lint-free, machine washable, vibrant colors."),
        ("Godrej Interio Office Chair", "Godrej", 8999, 14999,
         "Ergonomic design, adjustable armrest and height, lumbar support, 5-star base."),
        ("Home Centre Wooden Wall Shelf", "Home Centre", 1499, 2499,
         "3-tier wooden wall shelf, MDF material, easy installation, holds up to 10kg per tier."),
    ],
    "Books|Academic & Educational": [
        ("Python Crash Course by Eric Matthes", "No Starch Press", 599, 899,
         "Hands-on, project-based introduction to programming in Python for beginners."),
        ("Clean Code by Robert Martin", "Prentice Hall", 699, 1099,
         "Handbook of agile software craftsmanship. A must-read for every developer."),
        ("Machine Learning by Tom Mitchell", "McGraw Hill", 899, 1299,
         "Classic textbook on ML concepts, algorithms, and applications. Academic standard."),
        ("Deep Learning by Goodfellow et al", "MIT Press", 1299, 1899,
         "Comprehensive coverage of deep learning theory and practice by leading researchers."),
        ("Designing Data-Intensive Applications", "O Reilly Media", 999, 1499,
         "Big ideas behind reliable, scalable, and maintainable systems. Essential reading."),
        ("The Pragmatic Programmer 20th Anniversary", "Pragmatic Bookshelf", 899, 1299,
         "Your journey to mastery. Covers career, coding, and project practices."),
        ("Introduction to Algorithms CLRS", "MIT Press", 1499, 2199,
         "Comprehensive textbook covering algorithms, data structures, and complexity."),
    ],
    "Sports & Outdoors|Fitness Equipment": [
        ("Decathlon Domyos Fitness Mat", "Decathlon", 999, 1799,
         "7mm thick, non-slip texture, PVC free, foldable, 180cm x 60cm, yoga and pilates."),
        ("Cosco Gym Ball 75cm", "Cosco", 699, 1299,
         "Anti-burst technology, 200kg weight capacity, includes foot pump, PVC free."),
        ("Boldfit Adjustable Dumbbell Set", "Boldfit", 4999, 7999,
         "20kg total weight, adjustable plates, ergonomic grip, chrome finish, storage stand."),
        ("TRX Suspension Training System", "TRX", 7999, 12999,
         "Military-grade nylon, adjustable straps, door anchor, carry bag, full body workout."),
        ("Reebok Jump Rope", "Reebok", 799, 1299,
         "Speed rope, ball bearing handles, adjustable length, 3mm PVC cable, tangle-free."),
        ("Nivia Storm Football Size 5", "Nivia", 599, 999,
         "TPU material, 32-panel design, machine stitched, suitable for training and match."),
        ("Cosco Cricket Bat English Willow", "Cosco", 2499, 3999,
         "English willow grade 3, 5-6 grains, full cane handle, ready to play."),
    ],
    "Beauty & Personal Care|Skincare": [
        ("Minimalist Niacinamide 10 Percent Serum", "Minimalist", 399, 599,
         "10% niacinamide with 0.1% zinc, controls sebum, minimizes pores, brightens skin."),
        ("Dot and Key Barrier Repair Moisturizer", "Dot and Key", 699, 999,
         "Ceramide-rich, hyaluronic acid, centella asiatica, repairs damaged skin barrier."),
        ("Plum Bright Years Cell Renewal Serum", "Plum", 849, 1299,
         "0.3% retinol, bakuchiol, reduces fine lines, cell renewal, suitable for all skin types."),
        ("Mamaearth Vitamin C Face Wash", "Mamaearth", 299, 449,
         "Vitamin C + Turmeric, brightens skin, removes tan, gentle for daily use, toxin-free."),
        ("Himalaya Purifying Neem Face Wash", "Himalaya", 175, 225,
         "Neem extract, blackhead removal, oil control, suitable for oily and combination skin."),
        ("Lakme Absolute Skin Natural Mousse SPF 8", "Lakme", 349, 499,
         "SPF 8 PA+ mousse foundation, 24hr moisture, sweat-proof, buildable coverage."),
    ],
    "Beauty & Personal Care|Hair Care": [
        ("Pantene Advanced Hairfall Solution Shampoo", "Pantene", 349, 499,
         "Pro-Vitamin formula, anti-hairfall, strengthens hair, suitable for all hair types."),
        ("L Oreal Paris Extraordinary Oil Serum", "L Oreal Paris", 549, 799,
         "6 precious flower oils, 24hr shine, frizz control, lightweight non-greasy formula."),
        ("Dove Intense Repair Conditioner 700ml", "Dove", 399, 549,
         "Keratin actives, repairs damage, reduces breakage, suitable for damaged hair."),
        ("Tresemme Keratin Smooth Mask", "Tresemme", 299, 449,
         "Keratin and argan oil, deep conditioning, frizz control, 3 minute treatment."),
        ("Mamaearth Onion Hair Oil 150ml", "Mamaearth", 249, 399,
         "Red onion extract, redensyl, fights hairfall, promotes growth, mineral oil free."),
    ],
    "Appliances|Large Appliances": [
        ("Whirlpool 1.5 Ton 3 Star Split AC", "Whirlpool", 32999, 44999,
         "3-star BEE rating, 1.5 ton capacity, 5-in-1 convertible, IntelliSense inverter."),
        ("LG 260L 3 Star Double Door Refrigerator", "LG", 28999, 38999,
         "260 liter capacity, Smart Inverter Compressor, Moist Balance Crisper, door cooling."),
        ("Samsung 8kg Front Load Washing Machine", "Samsung", 34999, 49999,
         "8kg capacity, 1400 RPM, Digital Inverter Motor, Eco Bubble technology, WiFi enabled."),
        ("Voltas 1.5 Ton 5 Star Window AC", "Voltas", 29999, 39999,
         "5-star BEE rating, 1.5 ton, 100% copper condenser, anti-dust filter, auto restart."),
        ("Bosch 6kg Front Load Washing Machine", "Bosch", 38999, 54999,
         "6kg capacity, 1200 RPM, EcoSilence Drive, ActiveWater Plus, 15 wash programs."),
    ],
    "Computers & Accessories|Peripherals": [
        ("Logitech MX Master 3S Mouse", "Logitech", 8999, 11999,
         "8K DPI sensor, MagSpeed wheel, ergonomic design, 70 day battery, multi-device."),
        ("Keychron K2 Mechanical Keyboard", "Keychron", 7999, 9999,
         "75% layout, Gateron Red switches, RGB backlight, Bluetooth 5.1, Mac and Windows."),
        ("Dell 27 inch 4K USB-C Monitor P2723QE", "Dell", 54999, 69999,
         "4K UHD IPS, 60Hz, USB-C 90W delivery, height-tilt-swivel-pivot adjustable stand."),
        ("Seagate Backup Plus Portable 2TB", "Seagate", 4999, 6999,
         "2TB USB 3.0, compact design, 3-year warranty, automatic backup software included."),
        ("TP-Link Archer AX73 WiFi 6 Router", "TP-Link", 12999, 17999,
         "WiFi 6, AX5400, 6 antennas, OFDMA, MU-MIMO, OneMesh compatible, parental controls."),
        ("Crucial P3 1TB NVMe SSD M.2", "Crucial", 5999, 8499,
         "PCIe Gen3 NVMe, 3500MB/s read, 3000MB/s write, 5-year warranty, compatible."),
    ],
    "Gaming|Gaming Consoles & Accessories": [
        ("Sony PlayStation 5 Disc Edition", "Sony", 49990, 54990,
         "AMD Zen 2 CPU, RDNA 2 GPU, 825GB SSD, 4K 120Hz, Ray Tracing, DualSense controller."),
        ("Microsoft Xbox Series X", "Microsoft", 49990, 54990,
         "12 teraflops GPU, 1TB NVMe SSD, 4K 120Hz, Quick Resume, backward compatible."),
        ("Nintendo Switch OLED Model", "Nintendo", 34999, 39999,
         "7 inch OLED screen, 64GB storage, wired LAN port, enhanced audio, adjustable stand."),
        ("Logitech G29 Driving Force Racing Wheel", "Logitech", 18999, 24999,
         "900 degree rotation, dual-motor force feedback, leather steering wheel, compatible."),
        ("Razer DeathAdder V3 HyperSpeed", "Razer", 7999, 9999,
         "Focus Pro 30K DPI sensor, HyperSpeed wireless, ergonomic design, 90hr battery."),
    ],
}


def generate_product_id(index: int) -> str:
    """Generate a realistic Kaggle-style alphanumeric product ID."""
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "B" + "".join(random.choices(chars, k=9))


def format_price(amount: int) -> str:
    """
    Format price as Indian Rupee string matching the Kaggle dataset format.
    Uses the Unicode rupee sign (Rs.) spelled out to avoid encoding issues
    on Windows consoles while still writing the correct symbol to the CSV file.
    """
    # The Indian Rupee sign (U+20B9) is used in the dataset.
    # We write it to the UTF-8 CSV; print() uses ASCII-safe representation.
    return f"\u20b9{amount:,}"


def format_discount(actual: int, discounted: int) -> str:
    """Calculate and format discount percentage string."""
    if actual <= 0:
        return "0%"
    discount = round((actual - discounted) / actual * 100)
    return f"{discount}%"


def generate_rating() -> str:
    """Generate a realistic product rating string."""
    rating = round(random.uniform(3.2, 4.8), 1)
    return str(rating)


def generate_rating_count() -> str:
    """Generate a realistic rating count string with commas."""
    count = random.randint(50, 50000)
    return f"{count:,}"


def generate_img_link(product_id: str) -> str:
    """Generate a placeholder Amazon-format image link."""
    return f"https://m.media-amazon.com/images/I/{product_id}._SX679_.jpg"


def generate_product_link(product_id: str) -> str:
    """Generate a placeholder Amazon product link."""
    return f"https://www.amazon.in/dp/{product_id}"


def safe_print(text: str) -> None:
    """
    Print text safely on any platform, replacing unencodable characters.
    Prevents UnicodeEncodeError on Windows PowerShell with default cp1252.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoded = text.encode(sys.stdout.encoding or "ascii", errors="replace")
        print(encoded.decode(sys.stdout.encoding or "ascii"))


def main() -> None:
    """Generate the synthetic fallback dataset and save to CSV."""
    safe_print("=" * 60)
    safe_print("  Generating Synthetic Fallback Dataset")
    safe_print("  WARNING: This is for development use ONLY")
    safe_print("=" * 60)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    product_index = 0

    for category_path, products in CATEGORIES.items():
        for product_name, brand, discounted, actual, description in products:
            product_index += 1
            product_id = generate_product_id(product_index)

            row = {
                "product_id": product_id,
                "product_name": product_name,
                "category": category_path,
                "discounted_price": format_price(discounted),
                "actual_price": format_price(actual),
                "discount_percentage": format_discount(actual, discounted),
                "rating": generate_rating(),
                "rating_count": generate_rating_count(),
                "about_product": description,
                "img_link": generate_img_link(product_id),
                "product_link": generate_product_link(product_id),
            }
            rows.append(row)

    if not rows:
        safe_print("ERROR: No products were generated. Check CATEGORIES dict.")
        sys.exit(1)

    # Write CSV with UTF-8 encoding (handles Rupee sign and other Unicode)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    safe_print(f"\nSUCCESS: Generated {len(rows)} products")
    safe_print(f"Output  : {OUTPUT_FILE}")
    safe_print("\nCategories included:")
    for cat, products in CATEGORIES.items():
        safe_print(f"  - {cat}: {len(products)} products")

    safe_print(
        "\n"
        "NEXT STEPS:\n"
        "  1. Set 'use_fallback: true' in src/config/config.yaml\n"
        "  2. Run: python -m src.pipelines.training_pipeline\n"
        "\n"
        "For production, use: python scripts/setup_dataset.py"
    )


if __name__ == "__main__":
    main()
