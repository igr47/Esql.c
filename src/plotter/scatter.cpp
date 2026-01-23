#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
#include <unordered_map>

namespace plt = matplot;

namespace Visualization {

    // Enhanced Scatter Plot with comprehensive styling
    void Plotter::plotScatter(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for scatter plot");
        }

        auto itX = data.numericData.begin();
        auto itY = std::next(data.numericData.begin());

        std::vector<double> xData = itX->second;
        std::vector<double> yData = itY->second;

        // DEBUG: Check what data we have
        std::cout << "DEBUG: Numeric columns count: " << data.numericData.size() << std::endl;
        std::cout << "DEBUG: Categorical columns count: " << data.categoricalData.size() << std::endl;
        
        // Check if we have a third column for colormap values (could be numeric OR categorical)
        bool hasColormapData = false;
        bool colormapFromCategorical = false;
        std::vector<double> colormapValues;
        std::vector<std::string> categoricalLabels;
        
        // First check for numeric colormap data (3rd numeric column)
        if (data.numericData.size() >= 3) {
            auto itColor = std::next(itY);
            colormapValues = itColor->second;
            hasColormapData = true;
            std::cout << "DEBUG: Using numeric colormap data from column: " << itColor->first << std::endl;
        } 
        // If no 3rd numeric column, check for categorical data that we can convert
        else if (!data.categoricalData.empty() && !config.style.colormap.empty()) {
            // Use the first categorical column for colormap
            auto catIt = data.categoricalData.begin();
            categoricalLabels = catIt->second;
            hasColormapData = true;
            colormapFromCategorical = true;
            std::cout << "DEBUG: Using categorical colormap data from column: " << catIt->first << std::endl;
        }

        std::vector<double> cleanX, cleanY, cleanColorValues;
        std::vector<std::string> cleanCategoricalLabels;
        
        for (size_t i = 0; i < xData.size(); ++i) {
            if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                cleanX.push_back(xData[i]);
                cleanY.push_back(yData[i]);
                
                if (hasColormapData) {
                    if (colormapFromCategorical) {
                        // Handle categorical data
                        if (i < categoricalLabels.size()) {
                            cleanCategoricalLabels.push_back(categoricalLabels[i]);
                        }
                    } else {
                        // Handle numeric data
                        if (i < colormapValues.size() && !std::isnan(colormapValues[i])) {
                            cleanColorValues.push_back(colormapValues[i]);
                        }
                    }
                }
            }
        }

        if (cleanX.empty() || cleanY.empty()) {
            throw std::runtime_error("No valid data points for scatter plot");
        }

        // If using categorical data for colormap, convert to numeric values
        if (colormapFromCategorical && !cleanCategoricalLabels.empty()) {
            std::unordered_map<std::string, double> categoryToValue;
            double currentValue = 0.0;
            
            for (const auto& label : cleanCategoricalLabels) {
                if (categoryToValue.find(label) == categoryToValue.end()) {
                    categoryToValue[label] = currentValue++;
                }
                cleanColorValues.push_back(categoryToValue[label]);
            }
            
            std::cout << "DEBUG: Converted " << categoryToValue.size() << " categories to numeric values" << std::endl;
        }

        // Create scatter plot - decide between colormap and single color
        auto ax = plt::gca();
        bool useColormap = !config.style.colormap.empty() && hasColormapData && !cleanColorValues.empty();
        
        std::cout << "DEBUG: useColormap = " << useColormap 
                  << ", colormap config = '" << config.style.colormap << "'"
                  << ", hasColormapData = " << hasColormapData
                  << ", cleanColorValues size = " << cleanColorValues.size() << std::endl;
        
        if (useColormap) {
            std::cout << "DEBUG: Entering colormap path!" << std::endl;
            
            // Set colormap FIRST on the axes
            std::string cmap = config.style.colormap;
            std::transform(cmap.begin(), cmap.end(), cmap.begin(), ::tolower);
            
            // Map colormap names
            if (cmap == "jet" || cmap == "parula") {
                ax->colormap(plt::palette::jet());
            } else if (cmap == "hot") {
                ax->colormap(plt::palette::hot());
            } else if (cmap == "cool") {
                ax->colormap(plt::palette::cool());
            } else if (cmap == "spring") {
                ax->colormap(plt::palette::spring());
            } else if (cmap == "summer") {
                ax->colormap(plt::palette::summer());
            } else if (cmap == "autumn") {
                ax->colormap(plt::palette::autumn());
            } else if (cmap == "winter") {
                ax->colormap(plt::palette::winter());
            } else if (cmap == "gray" || cmap == "grey") {
                ax->colormap(plt::palette::gray());
            } else if (cmap == "bone") {
                ax->colormap(plt::palette::bone());
            } else if (cmap == "copper") {
                ax->colormap(plt::palette::copper());
            } else if (cmap == "pink") {
                ax->colormap(plt::palette::pink());
            } else if (cmap == "lines") {
                ax->colormap(plt::palette::lines());
            } else if (cmap == "colorcube") {
                ax->colormap(plt::palette::colorcube());
            } else if (cmap == "prism") {
                ax->colormap(plt::palette::prism());
            } else if (cmap == "flag") {
                ax->colormap(plt::palette::flag());
            } else if (cmap == "white") {
                ax->colormap(plt::palette::white());
            } else if (cmap == "plasma") {
                ax->colormap(plt::palette::plasma());
            } else if (cmap == "viridis") {
                ax->colormap(plt::palette::viridis());
            } else if (cmap == "inferno") {
                ax->colormap(plt::palette::inferno());
            } else if (cmap == "magma") {
                ax->colormap(plt::palette::magma());
            }
            
            // Create sizes vector
            std::vector<double> sizes(cleanX.size(), config.style.markersize);
            
            // Create scatter WITH color data in constructor
            auto scatterObj = plt::scatter(cleanX, cleanY, sizes, cleanColorValues);
            
            // Enable filled markers for colormap visualization
            scatterObj->marker_face(true);
            
            // Apply marker style (but NOT marker_color!)
            std::string marker = parseMarker(config.style.marker);
            if (!marker.empty() && marker != "none") {
                scatterObj->marker_style(marker);
            }
            
            // Set line width
            scatterObj->line_width(config.style.linewidth);
            
            // Add colorbar
            plt::colorbar();
            
            std::cout << "DEBUG: Colormap applied successfully!" << std::endl;
            
        } else {
            std::cout << "DEBUG: Entering single color path." << std::endl;
            
            // Single color path
            std::vector<double> sizes(cleanX.size(), config.style.markersize);
            auto scatterObj = plt::scatter(cleanX, cleanY, sizes);
            
            // Apply marker styling
            std::string marker = parseMarker(config.style.marker);
            if (!marker.empty() && marker != "none") {
                scatterObj->marker_style(marker);
                
                auto markerColor = parseColor(config.style.markercolor);
                scatterObj->marker_color(markerColor);
                
                if (config.style.markerfacecolor != "none") {
                    auto markerFaceColor = parseColor(config.style.markerfacecolor);
                    scatterObj->marker_face_color(markerFaceColor);
                    scatterObj->marker_face(true);
                } else {
                    scatterObj->marker_face(false);
                }
            }
            
            scatterObj->line_width(config.style.linewidth);
            auto markerColor = parseColor(config.style.markercolor);
            markerColor[3] = config.style.alpha;
            scatterObj->marker_color(markerColor);
        }

        // Set axis labels
        if (!config.xLabel.empty()) {
            plt::xlabel(config.xLabel);
        }
        if (!config.yLabel.empty()) {
            plt::ylabel(config.yLabel);
        }
        if (!config.title.empty()) {
            plt::title(config.title);
        }

        // Apply grid if requested
        if (config.style.grid) {
            plt::grid(true);
            //ax->grid_style(parseLineStyle(config.style.gridstyle));
        }

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

} // namespace Visualization
