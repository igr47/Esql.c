#include "storagemanager.h"
#include <stdexcept>
#include <thread>
#include <sstream>

//=================Pager Class 	Implementation=================================
Pager::Pager(const std::string& fname) : filename(fname){
	file.open(filename,std::ios::binary | std::ios::in | std::ios::out);
	if(!file.open()){
		file.open(filename,std::ios::binary | std::ios::out);
		file.close();
		file.open(filename,std::ios::binary | std::ios::in |std::ios::out);
	}
}

//destructor
Pager::~Pager(){
	file.close();
}

void Pager::read_page(uint32_t page_id,Node* node){
	filoe.seekg(page_id * PAGE_SIZE,std::ios::beg);
	file.read(reinterpret_cast<char*>(node),PAGE_SIZE);
	if(!file)throw std::runtime_error("Failed to read page" +std::to_string(page_id));
}

void Pager::write_page(uint32_t page_id,Node* node){
	file.seekp(page_id * PAGE_SIZE,std::ios::beg);
	file.write(reinterpret_cast<const char*>(node),PAGE_SIZE);
	file.flush();
	if(!file)throw std::runtime_error("Failed to write page "+std::to_string(page_id));
}

uint32_t Pager::allocate_page(){
	file.seekg(0,std::ios::end);
	uint32_t page_id=file.tellp() / PAGE_SIZE;
	Node node={};
	write_page(page_id,&node);
	return page_id;
}

uint64_t Pager::write_data_block(const std::string& data){
	file.seekp(0,std::ios::end);
	uint64_t offset=file.teep();
	std::vector<char> compressed(ZSTD_compressBound(data.size()));
	size_t compressed_size=ZSTD_compress(compressed.data()compressed.size(),data.data(),data.size(),1);
	if(ZSTD_isError(compressed_size)){
		throw std::runtime_error("Zstd compression failed for data block");
	}

	file.write(compressed.data(),compressed_size);
	return offset;
}

std::string Pager::read_data_block(uint64_t offset,uint32_t length){
	file.seekg(offset,std::ios::beg);
	std::vector<char> compressed(length);
	file.read(compressed.data,length);
	std::vector<char> decompressed(ZSTD_get_DecompressedSize(compressed.data(),length));
	size_t decompressed_size=ZSTD_decompress(decompressed.data(),decompressed.size(),compressed.data(),length);
	if(ZSTD_isError(decompressed_size)){
		throw std::runtime_error ("Zstd decompression failed for data block");
	}
	return std::string(decompressed.data(),decompressed_size);
}

//==================WriteAheadLog Implementation=====================================================
WriteAheadLog::WriteAheadLog(const std::string& fname) : filename(fname){
	log_file.open(fname+".wal",std::ios::app);
	if(!log_file.is_open)throw std::runtime_error("Failed to open WAL");
}

WriteAheadLog::~WriteAheadLog(){
	log_file.close();
}

void WriteAheadLog::log_operation(const std::string& op,uint32_t page_id,const Node* node){
	std::lock_guard<std::mutex> lock(log_mutex);
	log_file.write(op.c_str(),op.size());
	log_file.write("/0",1);
	log_file.write(reinterpret_cast<const cahr*>(&page_id),sizeof(page_id));
	if(node){
		log_file.write(reinterpret_cast<const char*>(node),PAGE_SIZE);
	}
	log_file.flush();
}

void WriteAheadLog::checkpoint(Pager& pager,uint32_t metadata_page_id){
	log_operation("CHECKPOINT",metadata_page_id,nullptr);
}

void WriteAheadLog::recover(Pager& pager,uint32_t metadata_page_id){
	log_file.seekg(0,std::ios::beg);
	uint32_t last_checkpoint_page_id=0;
	while(log_file){
		char op_buf[16];
		log_file.read(op_buf,16);
		std::string op(op_buf,strlen(op_buf,16));
		if(op.empty())break;
		uint32_t page_id;
		log_file.read(reinterpret_cast<char*>(&page_id),sizeof(page_id));
		if(op=="CHECKPOINT"){
			last_checkpoint_page_id=page_id;
			continue;
		}
		Node node;
		log_file.read(reinterpret_cast<char*>(node),PAGE_SIZE);
		if(log_file){
			pager.write_page(page_id,&node);
		}
	}
	metadata_page_id=last_checkpoint_page_id;
	log_file.close();
	std::remove(filename+".wal").c_str();
}

//=================BufferPool Imnplementation==============================================
//Caches frequently used data in memory to increase performance
//Uses Numa memory for Linux to increase performance

BufferPool::BufferPool(size_t max_size) : max_pages(max_size){
#ifdef __linux__
	if(numa_available()>=0){
		numa.set_prefered(-1);
	}
#endif
}

Node* BufferPool::get_page(Pager& pager,uint32_t page_id){
	std::lock_guard<std::mutex> lock(cache_mutex);
	if(cache.find(page_id) != cache.end()){
		return &cache[page_id];
	}
	Node node;
	pager.read_page(page_id,&node);
	if(cache.size()>=max_pages){
		cache.erase(cache.begin());
	}
	cache[page_id]=node;
	return &cache[page_id];
}

void BufferPool::write_page(Pager& pager,WriteAheadLog& wal,uint32_t page_id,const Node* node){
	std::lock_guard<std::mutex> lock(cache_mutex);
	cache[page_id]=*node;
	wal.log_operation("WRITE",page_id,node);
	pager.write_page(page_id,node);
}

void BufferPool::evict_page(uint32_t page_id){
	std::lock_guard<std::mutex> lock(cache_mutex);
	cache.erase(page_id);
}

void BufferPool::flush_all(){
	std::lock_guard<std::mutex> lock(cache_mutex);
	cache.clear();
}

