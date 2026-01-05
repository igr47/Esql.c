#include "plotter_includes/plotter.h"
#include "execution_engine_includes/executionengine_main.h"

ExecutionEngine::ResultSet ExecutionEngine::executePlot(Visualization::PlotStatement& stmt) {
    // Execute the underlying query
    auto queryResult = executeSelect(*stmt.query);

    // Create plotter
    Visualization::Plotter plotter;

    // Convert result to plot data
    auto plotData = plotter.convertToPlotData(queryResult);

    // Execute plot based on type
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
                default:
                    plotter.autoPlot(plotData, stmt.config.title);
                    break;
            }
    }

    // Save or show plot
    if (!stmt.config.outputFile.empty()) {
        plotter.savePlot(stmt.config.outputFile);
	ResultSet result;
        result.columns = {"status"};
        result.rows = {{"Plot saved to " + stmt.config.outputFile}};
        return result;
        //return ResultSet({"status"}, {{"Plot saved to " + stmt.config.outputFile}});
    } else {
        plotter.showPlot();
	ResultSet result;
        result.columns = {"status"};
        result.rows = {{"Plot displayed"}};
        return result;
        //return ResultSet({"status"}, {{"Plot displayed"}});
    }
}
