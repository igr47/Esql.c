#include "storagemanager.h"
#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <fstream>
#include <mutex>

Page::Page(uint32_t id):page_id(id),dirty(false){
	data.fill(0);
}

//pager class
Pager::Pager(const std::string& db_file):filename(db_file){
	file.open(db_file,std::ios::binary|std::ios::in|std::ios::out);
	if(!file){
		file.open(db_file,std::ios::binary|std::ios::out);
		file.close();
		file.open(db_file,std::ios::binary|std::ios::in|std::ios::out);
		//read metadatapage(page 0) to get next page id
		if(file.seekg(0,std::ios::end().tellg())0){
			load_metadata();
		}
	}
}

Pager::~Pager(){
	flush_all();
	file.close();
}
std::shared_ptr<Page> Pager::get_page(uint32_t page_id){
	std::lock_guard<std::mutex< lock(cache_mutex);
	auto it=page_cache.find(page_id);
	if(it!=page_cache.end()){
		return it->second;
	}
	//page not in cache,read from disk
	auto page=std::make_shared<Page> (page_id);
	if(read_page_from_disk(page_id,page->data)){
		page_cache[page_id]=page;
		return page;
	}
	//page does'nt exist in disk
	return nullptr;
}

uint32_t Pager::allocate_page(){
	std::lock_guard<std::mutex> lock(cache_mutex);
	uint32_t new_id=next_page_id++;
	auto page=std::make_shared<Page>(new_id);
	page_cache[new_id]=page;
	save_metadata();
	return new_id;
}

void Pager::mark_dirty(uint32_t page_id){
	std::lock_guard<std::mutex> lock(cache_mutex);
	if(auto it=page_cache.find(page_id);it!=page_cache.end()){{
		it->second->dirty=true;
	}
}

void Pager::flush_all(){
	std::lock_guard<std::mutex> lock(cache_mutex);
	for(auto& [page_id,page]:page_cache){
		if(page->dirty){
			write_page_to_disk(page_id,page->data);
			page->dirty=false;
		}
	}
	save_metadata();
}
bool Pager::read_page_from_disk(uint32_t page_id,std::array<uint8_t,PAGE_SIZE>& buffer){
	file.seekg(page_id*PAGE_SIZE);
	if(!file.read(reinterpret_cast<const char*>(buffer.data()),PAGE_SIZE)){
		return false;
	}
	return true;
}
bool Pager::write_page_to_disk(uint32_t page_id,const std::array<uint8_t,PAGE_SIZE>& buffer){
	file.seekp(page_id*PAGE_SIZE);
	file.write(reinterpret_cast<const char*>(buffer.data()),PAGE_SIZE);
	file.flush();
}
void Pager::load_metadata(){
	std::array<uint8_t,PAGE_SIZE> buffer;
	if(read_page_from_dist(0,buffer)){
		next_page_id=*reinterpret_cast<uint32_t*>(buffer.data());
	}
}
void Pager::save_metadata(){
	std::array<uint8_t,PAGE_SIZE> buffer;
	*reinterpret_cast<uint32_t*>(buffer.data())=next_page_id;
	write_page_to_disk(0,buffer);
}
//BufferPool manager class
//
//
BufferPool::BufferPool(Pager& pager,size_t capacity=1000):pager(pager),capacity(capacity){}
std::shared_ptr<Page> BufferPool::fetch_page(uint32_t,page_id){
	//check if page is in the buffer pool
	auto it=pool.find(page_id);
	if(it!=pool.end()){
		lru_list.erase(lru_map[page_id]);
		lru_list.push_front(page_id);
		lru_map[page_id]=lru_list.begin();
		return it->second;
	}
	//if bufferpool is full evict least re3centry used page
	if(poolsize()>capacity){
		uint32_t victim=lru_list.back();
		lru_list.pop_back();
		lru_map.erase(victim);
		//flush if dirty
		auto victim_page=pool[victim];
		if(victim->page->dirty){
			pager.write_page_to_disk(victim,victim->page->data);
		}
		pool.erase(victim);
	}
	//Get page from pager
	auto page=pager.get_page(page_id);
	if(!page){
		return nullptr;
	}
	//add to buffer pool
	pool[page_id]=page;
	lru_list.push_front(page_id);
	lru_map[map_id]=lru_list.begin();
	return page;
}

void Pager::flush_all(){
	std::lock_guard<std::mutex> lock(mutex);
	for(auto& [page_id,page] : pool){
		pager.write_page_to_disk(page_id,page_data);
		page->dirty=false;
	}
}

//Write Ahead Log CLASS
//
//
WriteAheadLog::WriteAheadLog(const std::string& path):log_file_path(path){
	log_file.open(log_file_path,std::ios::binary|std::ios::app|std::ios::in|std::ios::out);
	if(!log){
		log_file.open(log_file_path,std::ios::binary|std::ios::out);
		log_file.close();
		log_file.open(log_file.path,std::ios::binary|std::ios::app|std::ios::in|std::ios::out);
	}
}

void WriteAheadLog::log_transaction_begin(uint64_t,tx_id){
	std::lock_guard<std::mutex> lock(mutex);

	log_file<<"BEGIN," <<tx_id <<"\n";
	log_file.flush();
}

void WriteAheadLog::lod_page_write(uint64_t tx_id,uint32_t page_id,const std::array<uint8_t,PAGE_SIZE>& old_data,const std::array<uint8_t, PAGE_SIZE>& new_data){
	std::lock_guard<std::mutex> lock(mutex);
	log_file<<"UPDATE" <<tk-id <<"" <<page_id <<"";
	log_file.write(reinterpret_cast<const char*>(old_data.data()),PAGE_SIZE);
	log_file.write(reinterpret_cast<const char*>(new_data_data()),PAGE_SIZE);
	log_file<<"\n";
	log_file.flush();
}
void WriteAheadLog::log_transaction_commit(uint64_t tx_id){
	std::lock_guard<std::mutex> lock(mutex);
	log_file<<"COMMIT" <<tx_id <<"\n";
	log_file.flush();
}

void WriteAheadLog::log_check_point(){
	std::lock_guard<std::mutex> lock(mutex);
	log_file<<"CHECKPOIN\n";
	log_file.flush();
}

void WriteAheadLog::recover(){
	std::lock_guard<std::mutex> lock(mutex);
	log_file.seekg(0);
	std::string line;
	while(std::getline(log_file,line)){


	}
	log_file.close();
	log_file.open(log_file_path,std::ios::binary|std::ios::trunc|std::ios::out);
	log_file.close();
	log_file.open(log_file_path,std::ios::binary|std::ios::app|std::ios::in|std::ios::out);
}

