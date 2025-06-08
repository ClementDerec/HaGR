
class Plot:
    
    def __init__(self,):
        pass
        
        
    def plot_smoothed_surface_with_max(x, y, z, title="Smoothed surface with max", xlabel="x", ylabel="y", zlabel="z", smooth=0.01):
        """Plot a smoothed surface and highlight the maximum z value, keeping values near the maximum.

        Args:
            x (list): x-coordinates.
            y (list): y-coordinates.
            z (list): z-coordinates.
            title (str, optional): Title of the plot. Defaults to "Smoothed surface with max".
            xlabel (str, optional): Label for the x-axis. Defaults to "x".
            ylabel (str, optional): Label for the y-axis. Defaults to "y".
            zlabel (str, optional): Label for the z-axis. Defaults to "z".
        """
        
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import Rbf
        import os
        import csv
        

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        # === 2. Déterminer le maximum global brut ===
        max_idx_raw = np.argmax(z)
        x_max_raw = x[max_idx_raw]
        y_max_raw = y[max_idx_raw]
        z_max_raw = z[max_idx_raw]

        print("\n🔹 Maximum brut (non lissé) :")
        print(f"   Mutation rate = {x_max_raw:.4f}, Crossover rate = {y_max_raw:.4f}, Score = {z_max_raw:.4f}")

        # === 3. Créer une grille régulière ===
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        # === 4. Appliquer le filtre RBF avec noyau quintic ===
        rbf = Rbf(x, y, z, function='quintic', smooth=smooth)
        zi_smooth = rbf(xi, yi)

        # === 5. Trouver le maximum global lissé ===
        zmax = np.nanmax(zi_smooth)
        max_idx = np.unravel_index(np.nanargmax(zi_smooth), zi_smooth.shape)
        x_max, y_max = xi[max_idx], yi[max_idx]

        print("\n🟢 Maximum après filtrage (RBF - quintic) :")
        print(f"   Mutation rate = {x_max:.4f}, Crossover rate = {y_max:.4f}, Score = {zmax:.4f}")

        # === 6. Affichage 3D ===
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(xi, yi, zi_smooth, cmap='viridis', alpha=0.9, edgecolor='none')
        ax.contour(xi, yi, zi_smooth, zdir='z', offset=zi_smooth.min(), cmap='viridis', linewidths=1)

        # Maximum filtré
        ax.scatter(x_max, y_max, zmax, color='red', s=80, label='')
        ax.text(x_max, y_max, zmax + 0.01, f"Maximum\n({x_max:.2f}, {y_max:.2f}, {zmax:.3f})", color='red')

        # Maximum brut
        #ax.scatter(x_max_raw, y_max_raw, z_max_raw, color='blue', s=60, label='Max brut')
        #ax.text(x_max_raw, y_max_raw, z_max_raw + 0.01, f"Brut\n({x_max_raw:.2f}, {y_max_raw:.2f}, {z_max_raw:.3f})", color='blue')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title('Surface 3D (RBF quintic)' + title)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.legend()

        # === 7. Enregistrer la figure ===
        output_dir = "plot"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"figure_max_{title}.png")
        plt.savefig(output_path)
        print(f"\n✅ Figure enregistrée : {title}")

        # === 8. Enregistrer les résultats des maxima dans un CSV ===
        csv_path = os.path.join("plot", f"maxima_{title}.csv")
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Type", xlabel, ylabel, zlabel])
            writer.writerow(["Brut", x_max_raw, y_max_raw, z_max_raw])
            writer.writerow(["Filtré (quintic)", x_max, y_max, zmax])

        print(f"✅ Coordonnées des minima enregistrées dans : {csv_path}")
        plt.show()


    def plot_smoothed_surface_with_min(x, y, z, title="Smoothed surface with min", xlabel="x", ylabel="y", zlabel="z",smooth=0.01):
        """Plot a smoothed surface and highlight the minimum z value, keeping values near the minimum.

        Args:
            x (list): x-coordinates.
            y (list): y-coordinates.
            z (list): z-coordinates.
            title (str, optional): Title of the plot. Defaults to "Smoothed surface with min".
            xlabel (str, optional): Label for the x-axis. Defaults to "x".
            ylabel (str, optional): Label for the y-axis. Defaults to "y".
            zlabel (str, optional): Label for the z-axis. Defaults to "z".
        """
        
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import Rbf
        import os
        import csv

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        # === 2. Déterminer le minimum global brut ===
        min_idx_raw = np.argmin(z)
        x_min_raw = x[min_idx_raw]
        y_min_raw = y[min_idx_raw]
        z_min_raw = z[min_idx_raw]

        print("\n🔹 Minimum brut (non lissé) :")
        print(f"   Mutation rate = {x_min_raw:.4f}, Crossover rate = {y_min_raw:.4f}, Score = {z_min_raw:.4f}")

        # === 3. Créer une grille régulière ===
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        # === 4. Appliquer le filtre RBF avec noyau quintic ===
        rbf = Rbf(x, y, z, function='quintic', smooth=smooth)
        zi_smooth = rbf(xi, yi)

        # === 5. Trouver le minimum global lissé ===
        zmin = np.nanmin(zi_smooth)
        min_idx = np.unravel_index(np.nanargmin(zi_smooth), zi_smooth.shape)
        x_min, y_min = xi[min_idx], yi[min_idx]

        print("\n🟢 Minimum après filtrage (RBF - quintic) :")
        print(f"   Mutation rate = {x_min:.4f}, Crossover rate = {y_min:.4f}, Score = {zmin:.4f}")

        # === 6. Affichage 3D ===
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(xi, yi, zi_smooth, cmap='viridis', alpha=0.9, edgecolor='none')
        ax.contour(xi, yi, zi_smooth, zdir='z', offset=zi_smooth.min(), cmap='viridis', linewidths=1)

        # Minimum filtré
        ax.scatter(x_min, y_min, zmin, color='red', s=80, label='Maximum')
        ax.text(x_min, y_min, zmin + 0.01, f"Maximum\n({x_min:.2f}, {y_min:.2f}, {zmin:.3f})", color='red')

        # Minimum brut
        #ax.scatter(x_min_raw, y_min_raw, z_min_raw, color='blue', s=60, label='Min brut')
        #ax.text(x_min_raw, y_min_raw, z_min_raw + 0.01, f"Brut\n({x_min_raw:.2f}, {y_min_raw:.2f}, {z_min_raw:.3f})", color='blue')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(f'Surface 3D (RBF quintic), {title}')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.legend()

        # === 7. Enregistrer la figure ===
        output_dir = "plot"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"figure_min_{title}.png")
        plt.savefig(output_path)
        print(f"\n✅ Figure enregistrée : {output_path}")

        # === 8. Enregistrer les résultats des minima dans un CSV ===
        csv_path = os.path.join("plot", f"minima_{title}.csv")
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Type", xlabel, ylabel, zlabel])
            writer.writerow(["Brut", x_min_raw, y_min_raw, z_min_raw])
            writer.writerow(["Filtré (quintic)", x_min, y_min, zmin])

        print(f"✅ Coordonnées des minima enregistrées dans : {csv_path}")

        plt.show()

        
        
    def get_plot_coordinates_from_list(data: list) -> list[list[float]]:
        
        """Lit un fichier json contenant les coordonnées des points repères et les retourne sous forme de listes de coordonnées 2D ou 3D.

        Args:
            data (list): Nom du fichier json contenant les coordonnées des points repères.

        Returns:
            list[list[float]]: Liste des coordonnées sous forme de [[x values], [y values]] pour 2D
                               ou [[x values], [y values], [z values]] pour 3D.
        """
        
        import numpy as np
        
        if len(data[0]) == 2:
            x_values = [entry[0][1] for entry in data]  # Extract 'depth' values as x
            y_values = [entry[-1] for entry in data]  # Extract fitness values as y
            return [np.array(x_values), np.array(y_values)]
        if len(data[0]) == 3:
            x_values = [entry[0][1] for entry in data]  # Extract 'depth' values as x
            y_values = [entry[1][1] for entry in data]  # Extract 'K' values as y
            z_values = [entry[-1] for entry in data]  # Extract fitness values as z
            return [np.array(x_values), np.array(y_values), np.array(z_values)]
        
        
    def get_plot_coordinates_from_json(json_file: str) -> list[list[float]]:
        """Lit un fichier json contenant les coordonnées des points repères et les retourne sous forme de listes de coordonnées 2D ou 3D.

        Args:
            json_file (str): Nom du fichier json contenant les coordonnées des points repères.

        Returns:
            list[list[float]]: Liste des coordonnées sous forme de [[x values], [y values]] pour 2D
                               ou [[x values], [y values], [z values]] pour 3D.
        """
        
        import numpy as np
        from utils import MyUtils
        
        data = MyUtils.read_json(json_file)
        if len(data[0]) == 2 :
            x_values = [entry[0][1] for entry in data]  # Extract 'depth' values as x
            y_values = [entry[-1] for entry in data]  # Extract fitness values as y
            return [np.array(x_values), np.array(y_values)]
        if len(data[0]) == 3 :
            x_values = [entry[0][1] for entry in data]  # Extract 'depth' values as x
            y_values = [entry[1][1] for entry in data]  # Extract 'K' values as y
            z_values = [entry[-1] for entry in data]  # Extract fitness values as z
            return np.array([x_values, y_values, z_values])
        
        
    def plot_2d_surafce(x, y, z, title = "2D surface", xlabel = "x", ylabel = "y", zlabel = "z") :
        """Plot a 2D surface

        Args:
            x (list): _description_
            y (list): _description_
            z (list): _description_
            title (str, optional): _description_. Defaults to "2D surface".
            xlabel (str, optional): _description_. Defaults to "x".
            ylabel (str, optional): _description_. Defaults to "y".
            zlabel (str, optional): _description_. Defaults to "z".
        """
        import matplotlib.pyplot as plt
        from scipy.interpolate import griddata
        import numpy as np

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        
        # Create a grid for the area
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate the z values on the grid
        zi = griddata((x, y), z, (xi, yi), method='linear')

        # Plot the surface
        surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', alpha=0.8)

        # Add scatter points for reference
        #ax.scatter3D(x, y, z, color='red', marker='o')

        # Add a color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        plt.show()
    
        
    def plot_1d_surface(x, y, title = "1D surface", xlabel = "x", ylabel = "y") :
        """Plot a 1D surface

        Args:
            x (list): _description_
            y (list): _description_
            title (str, optional): _description_. Defaults to "1D surface".
            xlabel (str, optional): _description_. Defaults to "x".
            ylabel (str, optional): _description_. Defaults to "y".
        """
        import matplotlib.pyplot as plt
        import numpy as np
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Find the maximum y value and its corresponding x
        max_idx = y.index(max(y)) if isinstance(y, list) else int(np.argmax(y))
        x_max = x[max_idx]
        y_max = y[max_idx]

        # Highlight the maximum point
        plt.scatter([x_max], [y_max], color='red', zorder=5)
        plt.text(x_max, y_max, f"Max\n({x_max:.2f}, {y_max:.2f})", color='red', verticalalignment='bottom')

        plt.show()
        
     