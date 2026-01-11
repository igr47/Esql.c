#ifndef SCANNER_H
#define SCANNER_H
#include <string>
#include <unordered_map>
#include <vector>

class Token{
	public:
		enum class Type{
			//Keywords
			SELECT,FROM,WHERE,AND,OR,NOT,UPDATE,SET,DROP,TABLE,DELETE,INSERT,INTO,ALTER,CREATE,ADD,RENAME,VALUES,BOOL,TEXT,INT,FLOAT,DATABASE,DATABASES,SHOW,USE,TABLES,TO,ROW,BULK,IN,COLUMN,BETWEEN,GROUP,BY,HAVING,ORDER,ASC,DESC,LIMIT,OFFSET,PRIMARY_KEY,NOT_NULL,AS,DISTINCT,UNIQUE,AUTO_INCREAMENT,DEFAULT,CHECK,IS_NOT,IS,NULL_TOKEN,STRUCTURE,
            // Conditionals
            CASE,WHEN,THEN,ELSE,END,ROUND,LOWER,UPPER,SUBSTRING,LIKE,STATS,

            // File oerations
            LOAD,INFILE,LOCAL,DELIMITER,HEADER,

            // Window Functions
            ROW_NUMBER, RANK, DENSE_RANK, NTILE,LAG, LEAD, FIRST_VALUE, LAST_VALUE,OVER, PARTITION, WITHIN,

            // Date Functions
            JULIANDAY, YEAR, MONTH, DAY,NOW, CURRENT_DATE, CURRENT_TIMESTAMP,
	    
	    TIME_SERIES,QQ_PLOT,RESIDUALS,ANIMATION,INTERACTIVE,DASHBOARD,GEO_MAP,CANDLESTICK,
	    PARALLEL_COORDINATES,RADAR,QUIVER,STREAMPLOT,
	    
	    // Output formats
	    PNG,PDF,SVG,EPS,JPG,JPEG,GIF,MP4,HTML,

	    // Style-related tokens
	    XLABEL,YLABEL,ZLABEL,X_LABEL,Y_LABEL,Z_LABEL,SERIES,SERIES_NAMES,FORMAT,OUTPUT_FORMAT,CONTROLS,
	    WIDGETS,SLIDER,DROPDOWN,CHECKBOX,BUTTON,LABEL,STEP,OPTIONS,
	    
	    // Plot-specific tokens
	    FOR_X,X,Y,FOR_TARGET,TIME_COLUMN,GROUP_COLUMN,BY_GROUP,ANIMATE,ANIMATION_COLUMN,FPS,FRAMES_PER_SECOND,DASHBOARD_LAYOUT,LAYOUT,TO_FILE,

            // String Functions
            SUBSTR, CONCAT, LENGTH,

	    
	    PLOT, TITLE, LINE, SCATTER, BAR, HISTOGRAM, BOXPLOT, PIE, HEATMAP,
	    AREA, STACKED_BAR, MULTI_LINE, VIOLIN, CONTOUR, SURFACE,
	    WIREFRAME, HISTOGRAM_2D,TREND, TIME,FOR_COLUMNS,

            // Join Keywords
            INNER, LEFT, RIGHT, FULL, OUTER, JOIN, ON,

            // CTE
            WITH,

            // Other Functions
            NULLIF, COALESCE,

            // Ai Functionality
            TRAIN, PREDICT, INFERENCE, MODEL, MODELS, USING, WITH_MODEL,
            FEATURES, TARGET, TEST_SPLIT, ITERATIONS, ACCURACY, PRECISION,
            RECALL, F1_SCORE, CONFUSION_MATRIX, SAVE_MODEL, LOAD_MODEL,
            DROP_MODEL, SHOW_MODELS, MODEL_METRICS, FEATURE_IMPORTANCE,
            HYPERPARAMETERS, EPOCHS, BATCH_SIZE, LEARNING_RATE, REGULARIZATION,
            CROSS_VALIDATION, INFER, FORECAST, ANOMALY_DETECTION, CLUSTER,
            CLASSIFY, REGRESS, EXPLAIN, SHAP_VALUES,QUALITY,MARKDOWN,SAVE,
            COMPREHENSIVE,METRICS,RANKING,

            AI_PREDICT, AI_PREDICT_CLASS, AI_PREDICT_VALUE, AI_PREDICT_PROBA,
            AI_PREDICT_CLUSTER, AI_PREDICT_ANOMALY, AI_EXPLAIN, AI_TRAIN,
            AI_MODEL_METRICS, AI_FEATURE_IMPORTANCE, AI_ANALYZE,

            // Model management
            CREATE_MODEL, DESCRIBE_MODEL, CREATE_OR_REPLACE, PIPELINE,CATEGORICAL,NUMERIC,
            CLASSIFICATION,REGRESSION,BINARY,MULTICLASS,CLUSTERING,

            // AI-specific keywords
            WITH_PROBABILITY, WITH_CONFIDENCE, WITH_EXPLANATION,
            THRESHOLD, CONFIDENCE_LEVEL, TOP_K,PROBABILITY,EXPLANATION,

             // Additional ML algorithms
            GRADIENT_BOOSTING, DECISION_TREE, KNN, NAIVE_BAYES,
            KMEANS, DBSCAN, ISOLATION_FOREST,

            // Model types
            LIGHTGBM, XGBOOST, CATBOOST, RANDOM_FOREST, LINEAR_REGRESSION,
            LOGISTIC_REGRESSION, NEURAL_NETWORK, SVM,
            DONT_SAVE,OUTPUT,PROBABILITIES,CONFIDENCE,DETAILED,IF,EXISTS,FOR,TOP,

            // Evaluation metrics
            MAE, MSE, RMSE, R2, AUC_ROC, AUC_PR,

            // Feature engineering
            SCALER, NORMALIZER, ONE_HOT_ENCODER, LABEL_ENCODER, PCA,
            FEATURE_SELECTION, FEATURE_EXTRACTION,

            // Time series
            TIMESERIES, ROLLING_MEAN, EXPONENTIAL_SMOOTHING,

            ANALYZE, DATA, EXTENDED, SECTIONS,
            BEGIN, BATCH,PARALLEL, ROLLBACK,
            SIMILARITY, RECOMMENDATION, OUTLIER, DISTRIBUTION, SUMMARY,
            DESCRIPTION, JSON, CHART,DESCRIBE,TYPE,CORRELATION,IMPORTANCE,STEPS,STOP,CONTINUE,

            // Statisticl fuctions
            STDDEV,VARIANCE,PERCENTILE_CONT,CORR,REGR_SLOPE,

            // IS operations
            IS_NULL,IS_NOT_NULL,IS_TRUE,IS_NOT_TRUE,IS_FALSE,IS_NOT_FALSE,
            // Auto generations
            GENERATE_DATE, GENERATE_DATE_TIME,GENERATE_UUID,DATE,DATETIME,UUID,MOD,
			//Identifier & Literals
			IDENTIFIER,STRING_LITERAL,NUMBER_LITERAL,DOUBLE_QUOTED_STRING,
			//conditiinals
			TRUE,FALSE,
			//Aggregate functions
			COUNT,SUM,AVG,MIN,MAX,
			//OPERATORS
			EQUAL,NOT_EQUAL,LESS,LESS_EQUAL,GREATER,GREATER_EQUAL,ASTERIST,PLUS,MINUS,
			//Panctuation
			COMMA,DOT,SEMICOLON,L_PAREN,R_PAREN,COLON,SLASH,
			//Special
			END_OF_INPUT,ERROR
		};
		Type type;
		std::string lexeme;
		size_t line;
		size_t column;
		Token(Type type,const std::string& lexeme,size_t line,size_t column);
};
class Lexer{
	public:
		explicit Lexer(const std::string& input);
		Token nextToken();
		size_t getPosition () const {return position;}
		size_t getLine () const {return line;}
		size_t getColumn () const {return column;}
		void saveState(size_t& savePos,size_t& saveLine, size_t& saveCol) const{
			savePos= position;
			saveLine = line;
			saveCol = column;
		}
		void restoreState(size_t savePos,size_t saveLine,size_t savecol){
			position = savePos;
			line = saveLine;
			column = savecol;
		}
	private:
		const std::string input;
		size_t position;
		size_t line;
		size_t column;
		std::unordered_map<std::string,Token::Type> keywords;

		//I initialize the keywords
		void initializeKeyWords();
		void skipWhitespace();
		Token readIdentifierOrKeyword(size_t tokenline,size_t tokencolumn);
		Token readNumber(size_t tokenline,size_t tokencolumn);
		Token readString(size_t tokenline,size_t tokencolumn);
		Token readOperatorOrPanctuation(size_t tokenline,size_t tokencolumn);
};
#endif

