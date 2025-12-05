package ru.practicum;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

public class Main {
    public static void main(String[] args) throws Exception {
        BufferedImage colorImage = ImageIO.read(new File("../va.png"));
        BufferedImage grayImage = toGrayScale(colorImage);
        double[][] matrix = imageToMatrix(grayImage);
        double[][] kernel = normalizedKernel(0,0,9, 9, 50.0);
        double[][] filteredMatrix = applyGaussianFilter(matrix, kernel);
        BufferedImage filteredImage = matrixToImage(filteredMatrix);
        ImageIO.write(grayImage, "png", new File("original.png"));
        ImageIO.write(filteredImage, "png", new File("filtered.png"));
    }

    public static BufferedImage toGrayScale(BufferedImage colorImage) {
        BufferedImage gray = new BufferedImage(
                colorImage.getWidth(),
                colorImage.getHeight(),
                BufferedImage.TYPE_BYTE_GRAY
        );
        gray.getGraphics().drawImage(colorImage, 0, 0, null);
        return gray;
    }

    public static double[][] imageToMatrix(BufferedImage image) {
        int w = image.getWidth(), h = image.getHeight();
        double[][] matrix = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                matrix[y][x] = (image.getRGB(x, y) & 0xFF) / 255.0;
            }
        }
        return matrix;
    }

    public static BufferedImage matrixToImage(double[][] matrix) {
        int h = matrix.length, w = matrix[0].length;
        BufferedImage image = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int value = (int) (matrix[y][x] * 255);
                int rgb = (value << 16) | (value << 8) | value;
                image.setRGB(x, y, rgb);
            }
        }
        return image;
    }

    public static double[][] gausKernel(double a, double b, double paramGaus, int n, int m) {
        double[][] gaus = new double[n][m];
        int x_center = n / 2;
        int y_center = m / 2;
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < m; y++) {
                gaus[x][y] = 1 / (2 * Math.PI * paramGaus) * Math.exp(-(Math.pow((x - x_center - a), 2) + Math.pow((y - y_center - b), 2)) / (2 * Math.pow(paramGaus, 2)));
            }
        }
        return gaus;
    }

    public static double[][] normalizedKernel(double a, double b, int n, int m, double paramGaus) {
        double[][] kernel = gausKernel(a, b, paramGaus, n, m);
        double sum = sum_matrix(kernel, n, m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                kernel[i][j] = kernel[i][j] / sum;
            }
        }
        return kernel;
    }

    public static double sum_matrix(double[][] kernel, int n, int m) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                sum += kernel[i][j];
            }
        }
        return  sum;
    }

    public static double[][] applyGaussianFilter(double[][] image, double[][] kernel) {
        int imageHeight = image.length;
        int imageWidth = image[0].length;
        int kernelHeight = kernel.length;
        int kernelWidth = kernel[0].length;
        int padY = kernelHeight / 2;
        int padX = kernelWidth / 2;
        double[][] result = new double[imageHeight][imageWidth];
        for (int y = padY; y < imageHeight - padY; y++) {
            for (int x = padX; x < imageWidth - padX; x++) {
                double sum = 0.0;

                for (int ky = 0; ky < kernelHeight; ky++) {
                    for (int kx = 0; kx < kernelWidth; kx++) {
                        int imageY = y + ky - padY;
                        int imageX = x + kx - padX;
                        sum += image[imageY][imageX] * kernel[ky][kx];
                    }
                }
                result[y][x] = sum;
            }
        }
        return result;
    }
}