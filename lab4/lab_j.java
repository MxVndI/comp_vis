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

        // Новый код: выполнение алгоритма Кэнни
        double[][] magnitude = new double[filteredMatrix.length][filteredMatrix[0].length];
        double[][] angle = new double[filteredMatrix.length][filteredMatrix[0].length];
        sobelOperator(filteredMatrix, magnitude, angle);

        double[][] suppressed = nonMaximumSuppression(magnitude, angle);
        BufferedImage finalEdges = doubleThresholdFiltering(suppressed);
        ImageIO.write(finalEdges, "png", new File("final_edges.png"));
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

    // Новые методы для алгоритма Кэнни:

    public static void sobelOperator(double[][] image, double[][] magnitude, double[][] angle) {
        int height = image.length;
        int width = image[0].length;
        double[][] GxKernel = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        double[][] GyKernel = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                double gxVal = 0;
                double gyVal = 0;

                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        double pixelValue = image[y + i - 1][x + j - 1];
                        gxVal += pixelValue * GxKernel[i][j];
                        gyVal += pixelValue * GyKernel[i][j];
                    }
                }

                magnitude[y][x] = Math.sqrt(gxVal * gxVal + gyVal * gyVal);

                if (gxVal != 0) {
                    double angleRad = Math.atan2(gyVal, gxVal);
                    double angleDeg = angleRad * (180.0 / Math.PI);

                    if (angleDeg < 0)
                        angleDeg += 180;

                    angle[y][x] = angleDeg;
                }
            }
        }
    }

    public static int getDirection(double angle) {
        if (angle < 0)
            angle += 180;

        double tg = (angle != 90) ? Math.tan(angle * Math.PI / 180.0) : Double.POSITIVE_INFINITY;

        if (0 <= angle && angle < 90) {
            if (tg < -2.414)
                return 0;
            else if (-2.414 <= tg && tg < -0.414)
                return 1;
            else if (-0.414 <= tg && tg < 0.414)
                return 2;
            else
                return 3;
        } else {
            if (tg > 2.414)
                return 4;
            else if (0.414 < tg && tg <= 2.414)
                return 5;
            else if (-0.414 <= tg && tg <= 0.414)
                return 6;
            else
                return 7;
        }
    }

    public static double[][] nonMaximumSuppression(double[][] magnitude, double[][] angle) {
        int height = magnitude.length;
        int width = magnitude[0].length;
        double[][] suppressed = new double[height][width];

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                double currentAngle = angle[y][x];
                double currentMagnitude = magnitude[y][x];
                int direction = getDirection(currentAngle);

                double neighbor1 = 0;
                double neighbor2 = 0;

                switch (direction) {
                    case 0:
                        neighbor1 = magnitude[y + 1][x];
                        neighbor2 = magnitude[y - 1][x];
                        break;
                    case 1:
                        neighbor1 = magnitude[y + 1][x - 1];
                        neighbor2 = magnitude[y - 1][x + 1];
                        break;
                    case 2:
                        neighbor1 = magnitude[y][x - 1];
                        neighbor2 = magnitude[y][x + 1];
                        break;
                    case 3:
                        neighbor1 = magnitude[y - 1][x - 1];
                        neighbor2 = magnitude[y + 1][x + 1];
                        break;
                    case 4:
                        neighbor1 = magnitude[y + 1][x];
                        neighbor2 = magnitude[y - 1][x];
                        break;
                    case 5:
                        neighbor1 = magnitude[y + 1][x - 1];
                        neighbor2 = magnitude[y - 1][x + 1];
                        break;
                    case 6:
                        neighbor1 = magnitude[y][x - 1];
                        neighbor2 = magnitude[y][x + 1];
                        break;
                    case 7:
                        neighbor1 = magnitude[y - 1][x - 1];
                        neighbor2 = magnitude[y + 1][x + 1];
                        break;
                }

                if (currentMagnitude >= neighbor1 && currentMagnitude >= neighbor2)
                    suppressed[y][x] = currentMagnitude;
                else
                    suppressed[y][x] = 0;
            }
        }
        return suppressed;
    }

    public static BufferedImage doubleThresholdFiltering(double[][] suppressedMagnitude) {
        int height = suppressedMagnitude.length;
        int width = suppressedMagnitude[0].length;
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        double maxGrad = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (suppressedMagnitude[y][x] > maxGrad)
                    maxGrad = suppressedMagnitude[y][x];
            }
        }

        double lowLevel = maxGrad * 0.03;
        double highLevel = maxGrad * 0.3;

        System.out.println("Максимальный градиент: " + maxGrad);
        System.out.println("Нижний порог: " + lowLevel);
        System.out.println("Верхний порог: " + highLevel);

        boolean[][] strongEdges = new boolean[height][width];
        boolean[][] weakEdges = new boolean[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double gradValue = suppressedMagnitude[y][x];
                int value = 0;

                if (gradValue >= highLevel) {
                    strongEdges[y][x] = true;
                    value = 255;
                } else if (gradValue >= lowLevel) {
                    weakEdges[y][x] = true;
                    value = 128;
                }

                int rgb = (value << 16) | (value << 8) | value;
                result.setRGB(x, y, rgb);
            }
        }

        BufferedImage finalResult = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                finalResult.setRGB(x, y, result.getRGB(x, y));
            }
        }

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                if (weakEdges[y][x]) {
                    boolean hasStrongNeighbor = false;

                    for (int dy = -1; dy <= 1 && !hasStrongNeighbor; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dy == 0 && dx == 0) continue;
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                                if (strongEdges[ny][nx]) {
                                    hasStrongNeighbor = true;
                                    break;
                                }
                            }
                        }
                    }

                    int value = hasStrongNeighbor ? 255 : 0;
                    int rgb = (value << 16) | (value << 8) | value;
                    finalResult.setRGB(x, y, rgb);
                }
            }
        }

        return finalResult;
    }
}