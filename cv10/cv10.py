import cv2
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nepodarilo sa načítať obrázok: {path}")
    return img

def match_images_sift(img1, img2):
    # Inicializácia SIFT detektora
    sift = cv2.SIFT_create()

    # Detekcia kľúčových bodov a výpočet deskriptorov
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Vykreslenie zhôd
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    print(f"Počet dobrých zhôd: {len(good_matches)}")
    return result_img

# Cesty k obrázkom
path1 = 'Samo_1.jpg'
path2 = 'Samo_4.jpg'

# Načítanie obrázkov
img1 = load_image(path1)
img2 = load_image(path2)

# Porovnanie obrázkov
matched_img = match_images_sift(img1, img2)

# Zobrazenie výsledku
plt.figure(figsize=(12, 6))
plt.imshow(matched_img)
plt.title('SIFT porovnanie')
plt.axis('off')
plt.show()
