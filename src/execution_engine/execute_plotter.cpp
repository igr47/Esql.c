#include "plotter_includes/plotter.h"
#include "execution_engine_includes/executionengine_main.h"

ExecutionEngine::ResultSet ExecutionEngine::executePlot(Visualization::PlotStatement& stmt) {
    // Execute the underlying query
    auto queryResult = executeSelect(*stmt.query);

    // Create plotter
    Visualization::Plotter plotter;

    // Convert result to plot data
    auto plotData = plotter.convertToPlotData(queryResult);

    // Execute plot based on type and subType
    switch (stmt.subType) {
        case Visualization::PlotStatement::PlotSubType::CORRELATION:
            plotter.plotCorrelationMatrix(plotData);
            break;
        case Visualization::PlotStatement::PlotSubType::DISTRIBUTION:
            if (!stmt.targetColumn.empty()) {
                plotter.plotDistribution(plotData, stmt.targetColumn);
            } else if (!plotData.numericData.empty()) {
                plotter.plotDistribution(plotData, plotData.numericData.begin()->first);
            }
            break;
        case Visualization::PlotStatement::PlotSubType::TREND:
            if (!stmt.xColumns.empty() && !stmt.yColumns.empty()) {
                plotter.plotTrendLine(plotData, stmt.xColumns[0], stmt.yColumns[0]);
            } else if (plotData.numericData.size() >= 2) {
                auto it = plotData.numericData.begin();
                std::string xCol = it->first;
                std::string yCol = (++it)->first;
                plotter.plotTrendLine(plotData, xCol, yCol);
            }
            break;
        case Visualization::PlotStatement::PlotSubType::TIME_SERIES:
            if (!stmt.timeColumn.empty() && !stmt.targetColumn.empty()) {
                plotter.plotTimeSeries(plotData, stmt.timeColumn, stmt.targetColumn, stmt.config);
            } else if (!stmt.xColumns.empty() && !stmt.yColumns.empty()) {
                plotter.plotTimeSeries(plotData, stmt.xColumns[0], stmt.yColumns[0], stmt.config);
            }
            break;
        case Visualization::PlotStatement::PlotSubType::QQ_PLOT:
            if (!stmt.targetColumn.empty()) {
                plotter.plotQQPlot(plotData, stmt.targetColumn);
            } else if (!plotData.numericData.empty()) {
                plotter.plotQQPlot(plotData, plotData.numericData.begin()->first);
            }
            break;
        case Visualization::PlotStatement::PlotSubType::RESIDUALS:
            if (!stmt.xColumns.empty() && !stmt.yColumns.empty()) {
                // Residuals plot is not implemented in plotter.cpp yet, fallback to trend line
                plotter.plotTrendLine(plotData, stmt.xColumns[0], stmt.yColumns[0]);
            }
            break;
        case Visualization::PlotStatement::PlotSubType::ANIMATION:
            if (!stmt.animationColumn.empty()) {
                std::vector<Visualization::PlotData> frames;
                // Group data by animation column to create frames
                // This is simplified - in reality you'd need to properly group the data
                frames.push_back(plotData);
                plotter.createAnimation(frames, stmt.config, stmt.animationFPS);
            }
            break;
        case Visualization::PlotStatement::PlotSubType::INTERACTIVE:
            plotter.createInteractivePlot(plotData, stmt.config);
            break;
        case Visualization::PlotStatement::PlotSubType::DASHBOARD: {
            // For dashboard, we need multiple datasets/configs
            // For now, just plot the single dataset
            std::vector<Visualization::PlotData> datasets = {plotData};
            std::vector<Visualization::PlotConfig> configs = {stmt.config};
            plotter.createDashboard(datasets, configs, stmt.dashboardRows, stmt.dashboardCols);
	}
            break;
        default:
            // Standard plot based on plot type
            switch (stmt.config.type) {
                case Visualization::PlotType::LINE:
                    plotter.plotLine(plotData, stmt.config);
                    break;
                case Visualization::PlotType::SCATTER:
                    plotter.plotScatter(plotData, stmt.config);
                    break;
                case Visualization::PlotType::BAR:
                    plotter.plotBar(plotData, stmt.config);
                    break;
                case Visualization::PlotType::HISTOGRAM:
                    plotter.plotHistogram(plotData, stmt.config);
                    break;
                case Visualization::PlotType::BOXPLOT:
                    plotter.plotBoxPlot(plotData, stmt.config);
                    break;
                case Visualization::PlotType::PIE:
                    plotter.plotPie(plotData, stmt.config);
                    break;
                case Visualization::PlotType::HEATMAP:
                    plotter.plotHeatmap(plotData, stmt.config);
                    break;
                case Visualization::PlotType::MULTI_LINE:
                    plotter.plotMultiLine(plotData, stmt.config);
                    break;
                case Visualization::PlotType::AREA:
                    plotter.plotArea(plotData, stmt.config);
                    break;
                case Visualization::PlotType::STACKED_BAR:
                    plotter.plotStackedBar(plotData, stmt.config);
                    break;
                case Visualization::PlotType::VIOLIN:
                    plotter.plotViolin(plotData, stmt.config);
                    break;
                case Visualization::PlotType::CONTOUR:
                    plotter.plotContour(plotData, stmt.config);
                    break;
                case Visualization::PlotType::SURFACE:
                    plotter.plotSurface(plotData, stmt.config);
                    break;
                case Visualization::PlotType::WIREFRAME:
                    plotter.plotWireframe(plotData, stmt.config);
                    break;
                case Visualization::PlotType::HISTOGRAM_2D:
                    plotter.plotHistogram2D(plotData, stmt.config);
                    break;
		case Visualization::PlotType::GEO_MAP:
		    plotter.plotGeoMap(plotData, stmt.config);
		    break;
                default:
                    plotter.autoPlot(plotData, stmt.config.title);
                    break;
            }
    }

    // Save or show plot based on output format
    if (!stmt.config.outputFile.empty()) {
        // Determine format from outputFormat enum
        std::string format;
        switch (stmt.outputFormat) {
            case Visualization::PlotStatement::OutputFormat::PNG:
                format = "png";
                break;
            case Visualization::PlotStatement::OutputFormat::PDF:
                format = "pdf";
                break;
            case Visualization::PlotStatement::OutputFormat::SVG:
                format = "svg";
                break;
            case Visualization::PlotStatement::OutputFormat::JPG:
                format = "jpg";
                break;
            case Visualization::PlotStatement::OutputFormat::GIF:
                format = "gif";
                break;
            case Visualization::PlotStatement::OutputFormat::MP4:
                format = "mp4";
                break;
            case Visualization::PlotStatement::OutputFormat::HTML:
                format = "html";
                break;
            default:
                format = "png"; // default
        }
        
        plotter.savePlot(stmt.config.outputFile, format);
        ExecutionEngine::ResultSet result;
        result.columns = {"status"};
        result.rows = {{"Plot saved to " + stmt.config.outputFile}};
        return result;
    } else {
        plotter.showPlot();
        ExecutionEngine::ResultSet result;
        result.columns = {"status"};
        result.rows = {{"Plot displayed"}};
        return result;
    }
}
