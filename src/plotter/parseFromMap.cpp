#include "plotter_includes/plotter.h"
#include <algorithm>
#include <map>
#include <iostream>



namespace Visualization {

    void PlotConfig::Style::parseFromMap(const std::map<std::string, std::string>& styleMap) {
        for (const auto& [key, value] : styleMap) {
            std::string lowerKey = key;
            std::transform(lowerKey.begin(), lowerKey.end(), lowerKey.begin(), ::tolower);

            if (lowerKey == "color") {
                color = value;
            } else if (lowerKey == "colors") {
                // Parse comma separated list of colors
                std::vector<std::string> colorList;
                size_t start = 0;
                size_t end = 0;

                while (end != std::string::npos) {
                    end = value.find(',', start);

                    // Extract the color
                    std::string singleColor;
		    if (end == std::string::npos) {
                        singleColor = value.substr(start);
                    } else {
                        singleColor = value.substr(start, end - start);
                    }

                    // Trim whitespace
                    size_t first = singleColor.find_first_not_of(" \t\n\r");
                    size_t last = singleColor.find_last_not_of(" \t\n\r");

                    if (first != std::string::npos && last != std::string::npos) {
                        singleColor = singleColor.substr(first, last - first + 1);

                        // Remove quotes if present
                        if (singleColor.size() >= 2 &&
                            ((singleColor[0] == '\'' && singleColor.back() == '\'') || (singleColor[0] == '"' && singleColor.back() == '"'))) {

				  singleColor = singleColor.substr(1, singleColor.size() - 2);
                        }

                        if (!singleColor.empty()) {
                            colorList.push_back(singleColor);
                        }
                    }

                    start = (end == std::string::npos) ? end : end + 1;
                }

                if (!colorList.empty()) {
                    colors = colorList;
                }
            } else if (lowerKey == "linewidth") {
                try { linewidth = std::stod(value); } catch (...) {}
            } else if (lowerKey == "linestyle") {
                linestyle = value;
            } else if (lowerKey == "marker") {
                marker = value;
            } else if (lowerKey == "xlabel") {
                xlabel = value;
            } else if (lowerKey == "ylabel") {
                ylabel = value;
            } else if (lowerKey == "markercolor") {
                markercolor = value;
            } else if (lowerKey == "markersize") {
                try { markersize = std::stod(value); } catch (...) {}
	    } else if (lowerKey == "alpha") {
                try { alpha = std::stod(value); } catch (...) {}
            } else if (lowerKey == "grid") {
                grid = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "legend") {
                legend = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "width") {
                try { figwidth = std::stod(value); } catch (...) {}
            } else if (lowerKey == "height") {
                try { figheight = std::stod(value); } catch (...) {}
            } else if (lowerKey == "bins") {
                try { bins = std::stoi(value); } catch (...) {}
            } else if (lowerKey == "xmin") {
                try { xmin = std::stod(value); } catch (...) {}
            } else if (lowerKey == "xmax") {
                try { xmax = std::stod(value); } catch (...) {}
            } else if (lowerKey == "ymin") {
                try { ymin = std::stod(value); } catch (...) {}
            } else if (lowerKey == "ymax") {
                try { ymax = std::stod(value); } catch (...) {}
            } else if (lowerKey == "title_fontsize") {
                try { title_fontsize = std::stod(value); } catch (...) {}
            } else if (lowerKey == "facecolor") {
                facecolor = value;
            } else if (lowerKey == "edgecolor") {
                edgecolor = value;
            } else if (lowerKey == "barwidth") {
                try { barwidth = std::stod(value); } catch (...) {}
            } else if (lowerKey == "xtick_rotation") {
		try { xtick_rotation = std::stod(value); } catch (...) {}
            } else if (lowerKey == "ytick_rotation") {
                try { ytick_rotation = std::stod(value); } catch (...) {}
            } else if (lowerKey == "colormap") {
                colormap = value;
            } else if (lowerKey == "annotate") {
                annotate = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "fmt") {
                fmt = value;
            } else if (lowerKey == "cumulative") {
                cumulative = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "density") {
                density = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "showfliers") {
                showfliers = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "show_kde") {
                show_kde = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "rug") {
                rug = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "interactive") {
                interactive = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "save_format") {
                save_format = value;
	    } else if (lowerKey == "colormap") {
                colormap = value;
            } else if (lowerKey == "markerfacecolor") {
                markerfacecolor = value;
            } else if (lowerKey == "fliermarker") {
                fliermarker = value;
            } else if (lowerKey == "fliersize") {
                try { fliersize = std::stod(value); } catch (...) {}
            } else if (lowerKey == "explode") {
                // Parse comma separated list of explode values
                std::vector<double> explodeList;
                size_t start = 0;
                size_t end = 0;

                while (end != std::string::npos) {
                    end = value.find(',', start);

		                        // Extract the value
                    std::string singleValue;
                    if (end == std::string::npos) {
                        singleValue = value.substr(start);
                    } else {
                        singleValue = value.substr(start, end - start);
                    }

                    // Trim whitespace
                    size_t first = singleValue.find_first_not_of(" \t\n\r");
                    size_t last = singleValue.find_last_not_of(" \t\n\r");

                    if (first != std::string::npos && last != std::string::npos) {
                        singleValue = singleValue.substr(first, last - first + 1);

                        try {
                            explodeList.push_back(std::stod(singleValue));
                        } catch (...) {
                            // Ignore invalid values
                        }
		    }

		    start = (end == std::string::npos) ? end : end + 1;
                }

                if (!explodeList.empty()) {
                    explode = explodeList;
                }
            } else if (lowerKey == "autopct") {
                autopct = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "startangle") {
                startangle = value;
            } else if (lowerKey == "shadow") {
                shadow = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "stacked") {
                stacked = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "baralign") {
                baralign = value;
            } else if (lowerKey == "histtype") {
                histtype = value;
            } else if (lowerKey == "whiskerwidth") {
		try { whiskerwidth = std::stod(value); } catch (...) {}
            } else if (lowerKey == "gridstyle") {
                gridstyle = value;
            } else if (lowerKey == "gridalpha") {
                try { gridalpha = std::stod(value); } catch (...) {}
            } else if (lowerKey == "gridcolor") {
                gridcolor = value;
            } else if (lowerKey == "legend_loc") {
                legend_loc = value;
            } else if (lowerKey == "legend_ncol") {
                try { legend_ncol = std::stoi(value); } catch (...) {}
            } else if (lowerKey == "legend_fontsize") {
                try { legend_fontsize = std::stod(value); } catch (...) {}
            } else if (lowerKey == "dpi") {
                try { dpi = std::stod(value); } catch (...) {}
            } else if (lowerKey == "zmin") {
                try { zmin = std::stod(value); } catch (...) {}
            } else if (lowerKey == "zmax") {
                try { zmax = std::stod(value); } catch (...) {}
            } else if (lowerKey == "tick_fontsize") {
                try { tick_fontsize = std::stod(value); } catch (...) {}
            } else if (lowerKey == "xlabel_fontsize") {
		 try { xlabel_fontsize = std::stod(value); } catch (...) {}
            } else if (lowerKey == "ylabel_fontsize") {
                try { ylabel_fontsize = std::stod(value); } catch (...) {}
            } else if (lowerKey == "azimuth") {
                try { azimuth = std::stod(value); } catch (...) {}
                view_init_set = true;
            } else if (lowerKey == "elevation") {
                try { elevation = std::stod(value); } catch (...) {}
                view_init_set = true;
            } else if (lowerKey == "confidence_interval") {
                try { confidence_interval = std::stod(value); } catch (...) {}
            } else if (lowerKey == "fps") {
                try { fps = std::stoi(value); } catch (...) {}
            } else if (lowerKey == "repeat") {
                repeat = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "toolbar") {
                toolbar = value;
            } else if (lowerKey == "bbox_inches_tight") {
                bbox_inches_tight = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "pad_inches") {
                try { pad_inches = std::stod(value); } catch (...) {}
            } else if (lowerKey == "projection") {
                projection = value;
            } else if (lowerKey == "map_style") {
                map_style = value;
            } else if (lowerKey == "coastline_color") {
                coastline_color = value;
            } else if (lowerKey == "coastline_width") {
                try { coastline_width = std::stod(value); } catch (...) {}
            } else if (lowerKey == "show_coastlines") {
                show_coastlines = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "show_countries") {
                show_countries = (value == "true" || value == "1" || value == "yes");
	    } else if (lowerKey == "show_states") {
                show_states = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "show_cities") {
                show_cities = (value == "true" || value == "1" || value == "yes");
            } else if (lowerKey == "region") {
                region = value;
            } else if (lowerKey == "scalebar") {
                scalebar = value;
            } else if (lowerKey == "north_arrow") {
                north_arrow = value;
            } else if (lowerKey == "lat_min") {
                try { lat_min = std::stod(value); } catch (...) {}
            } else if (lowerKey == "lat_max") {
                try { lat_max = std::stod(value); } catch (...) {}
            } else if (lowerKey == "lon_min") {
                try { lon_min = std::stod(value); } catch (...) {}
            } else if (lowerKey == "lon_max") {
                try { lon_max = std::stod(value); } catch (...) {}
            } else if (lowerKey == "bubble_min_size") {
                try { bubble_min_size = std::stod(value); } catch (...) {}
            } else if (lowerKey == "bubble_max_size") {
		try { bubble_max_size = std::stod(value); } catch (...) {}
            } else if (lowerKey == "bubble_scale") {
                bubble_scale = value;
            } else if (lowerKey == "colorbar_title") {
                colorbar_title = value;
            } else if (lowerKey == "auto_scale") {
                auto_scale = (value == "true" || value == "1" || value == "yes");
            }
	}
    }
}

