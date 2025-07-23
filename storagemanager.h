#ifndef STORAGE_MANAGER_H
#define STORAGE_MANAGER_H
#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <fstream>
#include <mutex>

constexpr uint32_t PAGE_SIZE=4096;

class Page{
	public:
		uint32_t page_id;
		bool dirty;
		std::array<uint32_t,PAGE_SIZE> data;
		Page(uint32_t id);
};

//Pager class
class Pager{
	private:
		std::string filename;
		std::fstream file;
		std::unordered_map<uint32_t,std::shared_ptr<Page>> page_cache;
		std::mutex cache_mutex;
		uint32_t next_page_id=0;
	public:
		explicit Pager(const std::string& db_file);
		~Pager();

		std::shared_ptr<Page> get_page(uint32_t page_id);
		uint32_t allocate_page();
		void mark_dirty(uint32_t page_id);
		void flush_all();
	private:
		bool read_page_from_disk(uint32_t page_id,std::array<uint8_t,PAGE_SIZE>& buffer);
		bool write_page_to_disk(uint32_t page_id,const std::array<uint8_t,PAGE_SIZE>& buffer);
		void load_metadata();
		void save_metadata();
};
//BUFFERPOOL class
class BufferPool{
	private:
		Pager& pager;
		size_t capacity;
		std::unordered_map<uint32_t,std::shared_ptr<Page>>pool;
		std::list<uint32_t> lru_list;
		std::unordered_map<uint32_t,std::list<uint32_t>::iterator>lru_map;
		std::mutex mutex;
	public:
		BufferPool(Pager& pager,size_t capacity=1000);
		std::shared_ptr<Page> fetch_page(uint32_t,page_id);
		void flush_all();
};
//Class WRITE AHEAD LOG
class WriteAheadLog{
	private:
		std::string log_file_path;
		std::fstream log_file;
		std::mutex mutex;
	public:
		explicit WriteAheadLog(const std::string& path);
		void log_transaction_begin(uint64_t,tx_id);
		void lod_page_write(uint64_t tx_id,uint32_t page_id,const std::array<uint8_t,PAGE_SIZE>& old_data,const std::array<uint8_t, PAGE_SIZE>& new_data);
		void log_transaction_commit(uint64_t tx_id);
		void log_check_point();
		void recover();
