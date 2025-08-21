#ifndef STORAGE_MANAGER_H
#define STORAGE_MANAGER_H

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstring>
#include <map>
#include <mutex>
#include <atomic>
#include <zstd.h>
#include <algorithm>
#include <queue>
#include <thread>
#include <immintrin.h>
#include <optional>

#ifdef __linux__
#include <numa.h>
#endif

const size_t PAGE_SIZE=16384; //16kb node
const size_t KEY_SIZE=8; //int64_t keys
const size_t BPLUSTREE_ORDER=1000; //Approximately 100 keys per node
const size_t MAX_KEY_PREFIX=4; //For prifix compression
const size_t MAX_MESSAGES=100; //Max messages per node
const size_t DATA_BLOCK_SIZE=4096; //Size of data block for variable length

//Page types
enum class PageType : uint8_t{
	INTERNAM,LEAF,DATABLOCK,METADATA
};

//Message types
enum class MessageType : uint32_t{
	INSERT,UPDATE,DELETE
};

struct Mesage{
	MessageType type;
	int64_t key;
	uint64_t value_offset;
	uint32_t value_length;
};

struct PageHeader{
	PageType type;
	uint32_t page_id;
	uint32_t parent_page_id;
	uint32_t num_keys;
	uint32_t num_messages;
	uint32_t next_page_id;
	uint32_t prev_page_id;
	uint32_t version;
};

struct Node{
	PageHeader header;
	char data[PAGE_SIZE-sizeof(PageHeader)];
}

class Pager{
	private:
		std::fstream file;
		std::string filename;
	public:
		Pager(const std::string& fname);
		~pager();

		void read_page(uint32_t page_id,Node* node);
		void write_page(uint32_t page_id,Node* node);
		uint32_t allocate_page();
		uint64_t write_data_block(const std::string& data);
		std::string read_data_block(uint64_t offset,uint32_t length);
};

//Wrie AheadLog class declaration
class WriteAheadLog{
	private:
		std::fstream log_file;
		std::mutex log_mutex;
		std::string filename;

	public:
		WriteAheadLog(const std::string& fname);
		~WriteAheadLog();

		void log_operation(const std::string& op,uint32_t page_id,const Node* node);
		void checkpoint(Pager& pager,uint32_t metadata_page_id);
		void recover(Pager& pager,uint32_t metadata_page_id);
};

class BufferPool{
	private:
		std::map<uint32_t,Node> cache;
		std::mutex cache_mutex;
		size_t max_pages;
	public:
		BufferPool(size_t max_size);
		Node* get_page(Pager& pager,uint32_t page_id);
		void write_page(Pager& pager,WriteAheadLog& wal,uint32_t page_id,const Node* node);
		void evict_page(uint32_t page_id);
		void flush_all();
};
