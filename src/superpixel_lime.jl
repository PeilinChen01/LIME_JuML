using Images
using TestImages
using ImageSegmentation
using ColorTypes
using Plots
using Random  # Importiere das Random-Paket für die zufällige Farbgenerierung
using DataStructures  # Importiere das DataStructures-Paket für die Frequenzzählung


# Lade ein Beispielbild
img = load("Image/Clownfish.jpg")
# img = Gray.(testimage("house"))

# Führe die Felzenszwalb-Superpixel-Segmentierung durch
segments = felzenszwalb(img, 10, 100)  # 10 ist der Schwellwert-Wert, 50 mindestgröße der Superpixel

# Funktion zum Erzeugen einer zufälligen Farbe basierend auf einem Seed-Wert
function get_random_color(seed)
    Random.seed!(seed)  # 
    rand(RGB{N0f8})  # Generiere eine zufällige Farbe
end

# Erzeuge eine Farbbild-Darstellung der Superpixel-Segmente
# Erstelle eine Zufallskarte für die Superpixel-Labels
superpixel_labels = labels_map(segments)  # Extrahiere die Labels der Superpixel
colors = [get_random_color(i) for i in 0:maximum(superpixel_labels)]  # Generiere zufällige Farben für jedes Label
seg_colored = [colors[label+1] for label in superpixel_labels]  # Wende die Farben auf die Superpixel an

# Visualisierung der Superpixel-Segmente
seg_img = reshape(seg_colored, size(img))  # Reshape der Segmentierung zu einem Bild
plot(heatmap(seg_img), title="Felzenszwalb Superpixel Segmentation", size=(800, 600))

function plot_label(label, superpixel_labels, img)
    # Erstelle eine Maske für das angegebene Label
    mask = superpixel_labels .== label

    # Erzeuge ein Farbbild, das nur das angegebene Label hervorhebt
    highlighted_img = copy(img)
    highlighted_img[.!(mask)] .= RGB(0.0, 0.0, 0.0)  # Setze alle Pixel, die nicht zum Label gehören, auf schwarz

    # Plotte das Ergebnis
    plot(highlighted_img, title="Superpixel Label $label", size=(800, 600))
end


# Wähle ein Label zum Plotten (zum Beispiel das erste Label)
label_to_plot = 2  # Beispiel: 0 ist das erste Label
plot_label(label_to_plot, superpixel_labels, img)

