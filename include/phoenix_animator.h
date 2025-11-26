#ifndef PHOENIX_ANIMATOR_H
#define PHOENIX_ANIMATOR_H

#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>

class PhoenixAnimator {
private:
    int terminal_width_;
    std::vector<std::string> phoenix_frame_;  // Single static frame
    std::vector<const char*> fire_colors_;
    
    void initialize_frame();
    
public:
    PhoenixAnimator(int terminal_width = 80);
    void animate_fire_effect(int duration_ms = 6000);  // Changed to run in current thread
    void draw_static_phoenix();
    std::string apply_gradient(const std::string& line, int frame);
    const std::vector<std::string>& get_current_frame() const { return phoenix_frame_; }

};

#endif
