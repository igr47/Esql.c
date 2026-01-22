#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <map>
#include <regex>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

namespace plt = matplot;
using json = nlohmann::json;

namespace Visualization {

    bool Plotter::isLatitudeColumn(const std::vector<std::string>& values) {
        if (values.empty()) return false;

        size_t validCount = 0;
        for (const auto& val : values) {
            try {
                double lat = std::stod(val);
                if (lat >= -90.0 && lat <= 90.0) {
                    validCount++;
                }
            } catch (...) {
                // Not a valid number
            }
        }

        return (static_cast<double>(validCount) / values.size()) > 0.8;
    }

    bool Plotter::isLongitudeColumn(const std::vector<std::string>& values) {
        if (values.empty()) return false;

        size_t validCount = 0;
        for (const auto& val : values) {
            try {
                double lon = std::stod(val);
                if (lon >= -180.0 && lon <= 180.0) {
                    validCount++;
                }
            } catch (...) {
                // Not a valid number
            }
        }

        return (static_cast<double>(validCount) / values.size()) > 0.8;
    }

    std::vector<double> Plotter::convertToLatitude(const std::vector<std::string>& values) {
        std::vector<double> latitudes;
        latitudes.reserve(values.size());

        for (const auto& val : values) {
            try {
                double lat = std::stod(val);
                if (lat >= -90.0 && lat <= 90.0) {
                    latitudes.push_back(lat);
                } else {
                    latitudes.push_back(std::numeric_limits<double>::quiet_NaN());
                }
            } catch (...) {
                latitudes.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }

        return latitudes;
    }

    std::vector<double> Plotter::convertToLongitude(const std::vector<std::string>& values) {
        std::vector<double> longitudes;
        longitudes.reserve(values.size());

        for (const auto& val : values) {
            try {
                double lon = std::stod(val);
                if (lon >= -180.0 && lon <= 180.0) {
                    longitudes.push_back(lon);
                } else {
                    longitudes.push_back(std::numeric_limits<double>::quiet_NaN());
                }
            } catch (...) {
                longitudes.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }

        return longitudes;
    }

    void Plotter::setupGeoAxes(const PlotConfig& config) {
        plt::figure(true);
        plt::figure()->size(config.style.figwidth * 100, config.style.figheight * 100);

        // Set up geographic projection
        auto ax = plt::gca();

        // Note: Matplot++ doesn't have built-in geographic projections like Cartopy
        // We'll use standard Cartesian coordinates and handle projections manually if needed
        
        // Set map bounds
        if (config.style.auto_scale) {
            // Auto-scale will be handled by data
        } else {
            ax->xlim({config.style.lon_min, config.style.lon_max});
            ax->ylim({config.style.lat_min, config.style.lat_max});
        }

        // Set title
        if (!config.title.empty()) {
            plt::title(config.title);
        }

        // Set labels
        if (!config.xLabel.empty()) {
            plt::xlabel(config.xLabel);
        } else if (!config.style.lonLabel.empty()) {
            plt::xlabel(config.style.lonLabel);
        }

        if (!config.yLabel.empty()) {
            plt::ylabel(config.yLabel);
        } else if (!config.style.latLabel.empty()) {
            plt::ylabel(config.style.latLabel);
        }

        // Set grid
        if (config.style.grid) {
            plt::grid(true);
        }

        // Set tick parameters
        if (config.style.xtick_rotation != 0.0) {
            plt::xtickangle(config.style.xtick_rotation);
        }

        if (config.style.ytick_rotation != 0.0) {
            plt::ytickangle(config.style.ytick_rotation);
        }
    }

    void Plotter::validateGeoData(const PlotData& data, const PlotConfig& config) {
        bool hasLatitude = !data.latitudeData.empty();
        bool hasLongitude = !data.longitudeData.empty();
        bool hasRegions = !data.regionData.empty();

        switch (config.type) {
            case PlotType::GEO_SCATTER:
            case PlotType::GEO_HEATMAP:
            case PlotType::GEO_BUBBLE:
            case PlotType::GEO_LINE:
            case PlotType::GEO_CONTOUR:
            case PlotType::GEO_FLOW:
                if (!hasLatitude || !hasLongitude) {
                    throw std::runtime_error("Geographic plots require latitude and longitude data");
                }
                break;

            case PlotType::GEO_CHOROPLETH:
            case PlotType::GEO_POLYGON:
                if (!hasRegions) {
                    throw std::runtime_error("Choropleth and polygon plots require region data");
                }
                break;

            case PlotType::GEO_MAP:
                // Basic map doesn't require data
                break;
                
            default:
                break;
        }
        
        // Validate coordinate ranges
        for (const auto& [colName, lats] : data.latitudeData) {
            for (double lat : lats) {
                if (!std::isnan(lat) && (lat < -90.0 || lat > 90.0)) {
                    throw std::runtime_error("Invalid latitude value in column " + colName + 
                                           ": " + std::to_string(lat));
                }
            }
        }
        
        for (const auto& [colName, lons] : data.longitudeData) {
            for (double lon : lons) {
                if (!std::isnan(lon) && (lon < -180.0 || lon > 180.0)) {
                    throw std::runtime_error("Invalid longitude value in column " + colName + 
                                           ": " + std::to_string(lon));
                }
            }
        }
    }

    void Plotter::addBasemap(const PlotConfig& config) {
        // Note: Matplot++ doesn't have built-in basemaps like Cartopy
        // We'll create a simple coastline/background

        if (config.style.show_coastlines) {
            // Add simple coastline approximation
            std::vector<double> coast_lons, coast_lats;

            // Simple world coastline (simplified)
            for (double lon = -180.0; lon <= 180.0; lon += 1.0) {
                coast_lons.push_back(lon);
                coast_lats.push_back(-90.0 + 180.0 * std::abs(std::sin(lon * M_PI / 180.0)));
            }

            auto coast = plt::plot(coast_lons, coast_lats);
            auto coastColor = parseColor(config.style.coastline_color);
            // Set color - Matplot++ uses color strings or RGB vectors
            coast->color(coastColor);
            coast->line_width(config.style.coastline_width);
        }

        // Add grid lines
        addGridLines(config);

        // Add map features if requested
        addMapFeatures(config);
    }

    void Plotter::addGridLines(const PlotConfig& config) {
        if (!config.style.grid) return;

        // Add latitude lines
        for (double lat = -90.0; lat <= 90.0; lat += 30.0) {
            std::vector<double> lons = {-180.0, 180.0};
            std::vector<double> lats = {lat, lat};
            auto line = plt::plot(lons, lats);
            line->color(parseColor(config.style.gridcolor));
            line->line_width(0.5);
            line->line_style(":");
            // Alpha is handled in color parsing
        }

        // Add longitude lines
        for (double lon = -180.0; lon <= 180.0; lon += 30.0) {
            std::vector<double> lons = {lon, lon};
            std::vector<double> lats = {-90.0, 90.0};
            auto line = plt::plot(lons, lats);
            line->color(parseColor(config.style.gridcolor));
            line->line_width(0.5);
            line->line_style(":");
            // Alpha is handled in color parsing
        }
    }

    void Plotter::addMapFeatures(const PlotConfig& config) {
        // This is a simplified implementation
        // In a real application, you would load shapefiles or use a proper GIS library

        if (config.style.show_countries) {
            // Add major country boundaries (simplified)
            // USA
            std::vector<double> usa_lons = {-125.0, -66.0, -66.0, -125.0, -125.0};
            std::vector<double> usa_lats = {25.0, 25.0, 49.0, 49.0, 25.0};
            auto usa = plt::plot(usa_lons, usa_lats);
            usa->color("gray");
            usa->line_width(0.5);

            // Europe outline
            std::vector<double> europe_lons = {-10.0, 40.0, 40.0, -10.0, -10.0};
            std::vector<double> europe_lats = {35.0, 35.0, 70.0, 70.0, 35.0};
            auto europe = plt::plot(europe_lons, europe_lats);
            europe->color("gray");
            europe->line_width(0.5);

            // Asia outline
            std::vector<double> asia_lons = {40.0, 180.0, 180.0, 40.0, 40.0};
            std::vector<double> asia_lats = {0.0, 0.0, 70.0, 70.0, 0.0};
            auto asia = plt::plot(asia_lons, asia_lats);
            asia->color("gray");
            asia->line_width(0.5);
        }

        if (config.style.show_cities) {
            // Add major cities (simplified)
            std::map<std::string, std::pair<double, double>> major_cities = {
                {"New York", {-74.0, 40.7}},
                {"London", {-0.1, 51.5}},
                {"Tokyo", {139.7, 35.7}},
                {"Sydney", {151.2, -33.9}},
                {"Cairo", {31.2, 30.0}}
            };

            for (const auto& [city, coords] : major_cities) {
                // Create vectors for coordinates
                std::vector<double> city_x = {coords.first};
                std::vector<double> city_y = {coords.second};
                auto point = plt::scatter(city_x, city_y);
                point->marker_size(20);
                point->marker_face(true);
                point->marker_face_color("red");
                point->color("white");
                point->line_width(1);
            }
        }
    }

    void Plotter::addScaleBar(const PlotConfig& config) {
        if (config.style.scalebar == "none") return;

        // Calculate scale based on map bounds and projection
        double scale_length_km = 1000.0; // Default 1000 km

        // Position in bottom right corner
        double bar_x = config.style.lon_max - 20.0;
        double bar_y = config.style.lat_min + 10.0;

        // Draw scale bar
        std::vector<double> bar_xs = {bar_x, bar_x + 10.0};
        std::vector<double> bar_ys = {bar_y, bar_y};

        auto scale_bar = plt::plot(bar_xs, bar_ys);
        scale_bar->color("black");
        scale_bar->line_width(3);

        // Add label
        std::string label = "1000 km";
        if (config.style.scalebar == "miles") {
            label = "621 miles";
        }

        plt::text(bar_x + 5.0, bar_y - 2.0, label);
    }

    void Plotter::addNorthArrow(const PlotConfig& config) {
        if (config.style.north_arrow == "none") return;

        // Position in top right corner
        double arrow_x = config.style.lon_max - 15.0;
        double arrow_y = config.style.lat_max - 10.0;

        // Draw simple north arrow
        std::vector<double> arrow_xs = {arrow_x, arrow_x, arrow_x - 2.0, arrow_x, arrow_x + 2.0, arrow_x};
        std::vector<double> arrow_ys = {arrow_y - 5.0, arrow_y, arrow_y - 2.0, arrow_y, arrow_y - 2.0, arrow_y - 5.0};

        auto arrow = plt::fill(arrow_xs, arrow_ys);
        arrow->color("black");

        // Add "N" label
        plt::text(arrow_x, arrow_y + 2.0, "N");
    }

    std::pair<double, double> Plotter::calculateMapBounds(const PlotData& data) {
        if (data.latitudeData.empty() || data.longitudeData.empty()) {
            // Default world bounds
            return std::make_pair(-180.0, 180.0); // lon_min, lon_max
        }

        // Find min/max from all latitude/longitude data
        double min_lat = 90.0, max_lat = -90.0;
        double min_lon = 180.0, max_lon = -180.0;

        for (const auto& [_, lats] : data.latitudeData) {
            for (double lat : lats) {
                if (!std::isnan(lat)) {
                    min_lat = std::min(min_lat, lat);
                    max_lat = std::max(max_lat, lat);
                }
            }
        }

        for (const auto& [_, lons] : data.longitudeData) {
            for (double lon : lons) {
                if (!std::isnan(lon)) {
                    min_lon = std::min(min_lon, lon);
                    max_lon = std::max(max_lon, lon);
                }
            }
        }

        // Add padding (10%)
        double lat_padding = (max_lat - min_lat) * 0.1;
        double lon_padding = (max_lon - min_lon) * 0.1;

        return std::make_pair(
            std::max(min_lon - lon_padding, -180.0),
            std::min(max_lon + lon_padding, 180.0)
        );
    }

    std::vector<std::string> Plotter::getTerrainPalette(int n) {
        // Terrain color palette (green/brown for land)
        std::vector<std::string> colors;

        for (int i = 0; i < n; ++i) {
            float t = static_cast<float>(i) / (n - 1);

            // Green to brown gradient
            int r, g, b;
            if (t < 0.5) {
                // Dark green to light green
                r = static_cast<int>(34 * (1 - t * 2) + 152 * (t * 2));
                g = static_cast<int>(139 * (1 - t * 2) + 251 * (t * 2));
                b = static_cast<int>(34 * (1 - t * 2) + 152 * (t * 2));
            } else {
                // Light green to brown
                t = (t - 0.5) * 2;
                r = static_cast<int>(152 * (1 - t) + 139 * t);
                g = static_cast<int>(251 * (1 - t) + 69 * t);
                b = static_cast<int>(152 * (1 - t) + 19 * t);
            }

            std::stringstream ss;
            ss << "#" << std::hex << std::setw(2) << std::setfill('0') << r
               << std::setw(2) << std::setfill('0') << g
               << std::setw(2) << std::setfill('0') << b;
            colors.push_back(ss.str());
        }

        return colors;
    }

    std::vector<std::string> Plotter::getTopographicPalette(int n) {
        // Topographic palette (blue for water, green/brown for land)
        std::vector<std::string> colors;

        for (int i = 0; i < n; ++i) {
            float t = static_cast<float>(i) / (n - 1);

            int r, g, b;
            if (t < 0.3) {
                // Deep blue to light blue (water)
                t = t / 0.3;
                r = static_cast<int>(0 * (1 - t) + 173 * t);
                g = static_cast<int>(0 * (1 - t) + 216 * t);
                b = static_cast<int>(139 * (1 - t) + 230 * t);
            } else if (t < 0.6) {
                // Light blue to green (coastal)
                t = (t - 0.3) / 0.3;
                r = static_cast<int>(173 * (1 - t) + 34 * t);
                g = static_cast<int>(216 * (1 - t) + 139 * t);
                b = static_cast<int>(230 * (1 - t) + 34 * t);
            } else {
                // Green to brown to white (mountains)
                t = (t - 0.6) / 0.4;
                if (t < 0.5) {
                    t = t * 2;
                    r = static_cast<int>(34 * (1 - t) + 139 * t);
                    g = static_cast<int>(139 * (1 - t) + 69 * t);
                    b = static_cast<int>(34 * (1 - t) + 19 * t);
                } else {
                    t = (t - 0.5) * 2;
                    r = static_cast<int>(139 * (1 - t) + 255 * t);
                    g = static_cast<int>(69 * (1 - t) + 255 * t);
                    b = static_cast<int>(19 * (1 - t) + 255 * t);
                }
            }

            std::stringstream ss;
            ss << "#" << std::hex << std::setw(2) << std::setfill('0') << r
               << std::setw(2) << std::setfill('0') << g
               << std::setw(2) << std::setfill('0') << b;
            colors.push_back(ss.str());
        }

        return colors;
    }

    std::vector<std::string> Plotter::getOceanPalette(int n) {
        // Ocean color palette (deep blue to light blue)
        std::vector<std::string> colors;

        for (int i = 0; i < n; ++i) {
            float t = static_cast<float>(i) / (n - 1);

            int r = static_cast<int>(0 * (1 - t) + 135 * t);
            int g = static_cast<int>(0 * (1 - t) + 206 * t);
            int b = static_cast<int>(139 * (1 - t) + 250 * t);

            std::stringstream ss;
            ss << "#" << std::hex << std::setw(2) << std::setfill('0') << r
               << std::setw(2) << std::setfill('0') << g
               << std::setw(2) << std::setfill('0') << b;
            colors.push_back(ss.str());
        }

        return colors;
    }

    double Plotter::calculateGreatCircleDistance(double lat1, double lon1, double lat2, double lon2) {
        // Convert to radians
        double phi1 = lat1 * M_PI / 180.0;
        double phi2 = lat2 * M_PI / 180.0;
        double delta_phi = (lat2 - lat1) * M_PI / 180.0;
        double delta_lambda = (lon2 - lon1) * M_PI / 180.0;

        // Haversine formula
        double a = std::sin(delta_phi / 2) * std::sin(delta_phi / 2) +
                   std::cos(phi1) * std::cos(phi2) *
                   std::sin(delta_lambda / 2) * std::sin(delta_lambda / 2);
        double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));

        // Earth radius in kilometers
        double R = 6371.0;

        return R * c;
    }

    std::pair<std::vector<double>, std::vector<double>> Plotter::latLonToMercator(const std::vector<double>& lats,
                                                                                 const std::vector<double>& lons) {
        std::vector<double> x, y;
        x.reserve(lats.size());
        y.reserve(lats.size());

        for (size_t i = 0; i < lats.size(); ++i) {
            // Mercator projection
            double lat_rad = lats[i] * M_PI / 180.0;
            double lon_rad = lons[i] * M_PI / 180.0;

            // Prevent division by zero at poles
            double lat_merc = std::log(std::tan(M_PI / 4.0 + lat_rad / 2.0));

            x.push_back(lon_rad);
            y.push_back(lat_merc);
        }

        return std::make_pair(x, y);
    }
}
