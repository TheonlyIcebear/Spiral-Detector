from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.cluster import DBSCAN
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from skimage import feature
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np, scipy, cv2


class ImageProcessor:
    def __init__(self, image_path, blur_sigma=2.0, edge_sigma=1.0, method='eigen'):
        self.image_path = image_path
        self.blur_sigma = blur_sigma
        self.edge_sigma = edge_sigma
        self.image = Image.open(self.image_path)
        self.image_array = np.array(self.image)
        self.c = 1+0j # Constant for complex transformations
        self.method = method


    def apply_edge_detection(self):

        gray = cv2.cvtColor(self.image_array, cv2.COLOR_BGR2GRAY)
        blurred = gaussian_filter(gray, sigma=self.blur_sigma)
        edges = feature.canny(blurred, sigma=self.edge_sigma)
        self.edges_array = edges
        self.edge_image = Image.fromarray(edges)

    def display_images(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(self.image, cmap='gray')
        ax[0].set_title("Original")
        ax[0].axis('off')

        ax[1].imshow(self.edge_image, cmap='gray')
        ax[1].set_title("Edges")
        ax[1].axis('off')

        plt.show()


    def complex_log(self, z, c):
        # Apply complex logarithm with a constant offset
        _z = z - c  # Shift the complex number by a constant to change center
        log_z = np.log(np.abs(_z)) + 1j * (np.angle(_z))  # Shift angle by pi to change orientation
        log_z += c  # Shift back by the same constant
        return log_z
    
    def complex_exp(self, z, c):
        # Apply complex exponential with a constant offset
        _z = z - c  # Shift the complex number by a constant to avoid exp(0)
        exp_z = np.exp(_z)
        exp_z += c  # Shift back by the same constant
        return exp_z
    
    def covariance_matrix(self, z):
        # Compute the covariance matrix of the complex points
        return np.cov(z.real, z.imag)
    
    def eigen_decomposition(self, cov_matrix):
        # Perform eigen decomposition to find principal directions
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        return eigenvalues, eigenvectors
    
    def plot_anisotropy_3d(self, z):
        c_x_values = np.linspace(-1, 1, 100)
        c_y_values = np.linspace(-1, 1, 100)
        
        c_values = np.array([[complex(x, y) for x in c_x_values] for y in c_y_values])
        best_c = None
        best_metric = -np.inf
        anisotropy_values = np.zeros((100, 100))

        for i, row in enumerate(c_values):
            for j, c in enumerate(row):
                log_z = self.complex_log(z, c)
                cov_matrix = self.covariance_matrix(log_z)
                eigenvalues, eigenvectors = self.eigen_decomposition(cov_matrix)
                points = np.column_stack([log_z.real, log_z.imag])

                anisotropy = np.max(eigenvalues) ** 2 / np.min(eigenvalues)
                major_axis = eigenvectors[:, np.argmax(eigenvalues)]
                minor_axis = eigenvectors[:, np.argmin(eigenvalues)]

                projections_major = points @ major_axis
                projections_minor = points @ minor_axis

                n = len(projections_major)
                skew = scipy.stats.skew(projections_major)
                kurt = scipy.stats.kurtosis(projections_major)  # Fisher's definition, normal = 0

                bc = (skew**2 + 1) / (kurt + 3 * (n-1)**2 / ((n-2)*(n-3)))
                l = 1.1 # Weight for skew/kurtosis in the metric, can be tuned based on experimentation


                metric = anisotropy * (1 - l * bc) # Check for high anisotropy, low skew/kurtosis, and low correlation between major and minor projections

                if metric > best_metric:
                    best_metric = metric
                    best_c = c
                
                anisotropy_values[i][j] = metric

        # Build meshgrid for plotting
        X, Y = np.meshgrid(c_x_values, c_y_values)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, anisotropy_values, cmap='plasma', edgecolor='none', alpha=0.9)

        self.c = best_c # Update the constant to the best found value for further transformations

        fig.colorbar(surf, ax=ax, shrink=0.5, label='Metric')
        ax.set_xlabel('Re(c)')
        ax.set_ylabel('Im(c)')
        ax.set_zlabel('Linearity Metric')
        ax.set_title('Linearity over complex c values')
        # Label showing the best c value on the figure
        ax.text2D(
            0.02,
            0.95,
            f"Best c: {best_c.real:.3f}{'+' if best_c.imag >= 0 else '-'}{abs(best_c.imag):.3f}j",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
        )

        plt.tight_layout()
        plt.show()



    def start(self):
        self.apply_edge_detection()
        self.display_images()
        print(np.max(self.edges_array))
        ys, xs = np.where(self.edges_array == True)
        xs -= self.edges_array.shape[1] // 2 # Center the coordinates around the middle of the image
        ys -= self.edges_array.shape[0] // 2 # Center the coordinates around the middle of the image

        xs = xs.astype(np.float64)
        ys = -ys.astype(np.float64) # Invert y-axis to match mathematical convention (positive y goes up)

        aspect_ratio = self.edges_array.shape[1] / self.edges_array.shape[0]
        xs /= np.float64(self.edges_array.shape[1] // 2) # Normalize to range [-1, 1]
        ys /= np.float64(self.edges_array.shape[0] // 2) # Normalize to range [-1, 1]

        print(f"Number of edge points: {len(xs)}")

        self.xs, self.ys = xs, ys

        z = self.xs + self.ys * 1j # Convert to complex numbers for easier analysis
        self.plot_anisotropy_3d(z) # Plot anisotropy over a range of complex constants
        self.log_z = self.complex_log(z, self.c) # Apply complex logarithm to the edge points

        mask = self.log_z.real > -2 # Filter out points with very negative real parts to focus on the main structure
        self.log_z = self.log_z[mask]
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))

        # Log-transformed points
        ax[0].scatter(self.log_z.real, self.log_z.imag, s=1)

        covariance = self.covariance_matrix(self.log_z)
        eigenvalues, eigenvectors = self.eigen_decomposition(covariance)

        x = np.column_stack((self.log_z.real, self.log_z.imag))
        x_rotated = x @ eigenvectors


        mean = np.mean(self.log_z)

        # Plot the eigenvectors
        for i in range(2):
            vec = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
            scale = np.sqrt(eigenvalues[i]) * 2
            start = mean
            end = mean + (vec[0] + 1j * vec[1]) * scale
            ax[0].plot([start.real, end.real], [start.imag, end.imag], 'g-', linewidth=2, label=f"Eigenvector {i+1}")

        princicapal_direction = eigenvectors[:, np.argmax(eigenvalues)]
        principal_slope = princicapal_direction[1] / princicapal_direction[0]

        print(f"Principal slope: {principal_slope}")

        y_intercept = mean.imag - principal_slope * mean.real

        start = (np.pi - y_intercept)/principal_slope
        end = (-np.pi - y_intercept)/principal_slope

        if self.method == 'eigen':
            x = np.linspace(start, end, 1000)
            y = principal_slope * x + y_intercept

            ax[0].plot(x, y, 'm-', linewidth=2, label="Principal Direction")
            
            eigen_z = x + 1j * y
            eigen_curve = self.complex_exp(eigen_z, self.c)

            ax[1].scatter(eigen_curve.real, eigen_curve.imag, s=1, c='m', label="Exp of principal direction")
        
        if self.method == 'ransac':
            # Line of best fit
            x = self.log_z.real.reshape(-1, 1)
            y = self.log_z.imag

            model = RANSACRegressor(LinearRegression())
            model.fit(x, y)
            
            ax[0].plot(x.reshape(-1, 1), model.predict(x.reshape(-1, 1)), "r-", linewidth=2, label="Best fit")

            x = np.linspace(start - 0.5, end + 0.5, 1000)

            complex_line = x.reshape(-1) + 1j * model.predict(x.reshape(-1, 1)).reshape(-1)
            complex_curve = self.complex_exp(complex_line, self.c) # Apply complex exponential to the line of best fit
            ax[1].scatter(complex_curve.real, complex_curve.imag, s=1, c='r', label="Exp of best fit")

        ax[0].set_title("Log-Transformed Edge Points")
        ax[0].set_xlabel("Real")
        ax[0].set_ylabel("Imaginary")
        ax[0].legend()
        ax[0].axis('equal')

        # Original edge points
        ax[1].scatter(self.xs, self.ys, s=1, c='b', label="Original Edge Points")
        ax[1].set_title("Edge Points")
        ax[1].set_xlabel("X")
        ax[1].set_ylabel("Y")
        ax[1].set_aspect('equal', adjustable='box')

        h, w = self.image_array.shape[:2]
        ax[2].imshow(self.image_array, cmap='gray')

        if self.method == 'ransac':
            complex_x = (complex_curve.real + 1) * (w / 2)
            complex_y = (1 - complex_curve.imag) * (h / 2)
            
            ax[2].scatter(complex_x, complex_y, s=1, c='r', label="Exp of best fit")

        if self.method == 'eigen':
            eigen_x = (eigen_curve.real + 1) * (w / 2)
            eigen_y = (1 - eigen_curve.imag) * (h / 2)

            ax[2].scatter(eigen_x, eigen_y, s=1, c='m', label="Exp of principal direction")
        
        ax[2].set_title("Edge Points with Exponential Curve")
        ax[2].set_xlabel("X")
        ax[2].set_ylabel("Y")
        ax[2].set_aspect('equal', adjustable='box')
        ax[2].legend()

        ax[3].scatter(x_rotated[:, 0], x_rotated[:, 1], s=1, c='c', label="Rotated Log Points")
        ax[3].set_title("Rotated Log-Transformed Points")
        ax[3].set_xlabel("Principal Component 1")
        ax[3].set_ylabel("Principal Component 2")
        ax[3].legend()
        ax[3].set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.show()

        

if __name__ == "__main__":
    processor = ImageProcessor("images\\c0191344-800px-wm.jpg", blur_sigma=3.0, edge_sigma=4.0, method='ransac')
    # Method can be 'eigen' for eigenvector-based line detection or 'ransac' for RANSAC-based line fitting
    processor.start()



