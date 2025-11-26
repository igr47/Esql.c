#include "phoenix_animator.h"

// PhoenixAnimator implementation
PhoenixAnimator::PhoenixAnimator(int terminal_width)
    : terminal_width_(terminal_width) {

    // Fire gradient colors - from coolest to hottest
    fire_colors_ = {
        "\033[38;5;196m",  // Bright Red
        "\033[38;5;202m",  // Orange Red
        "\033[38;5;208m",  // Orange
        "\033[38;5;214m",  // Dark Yellow
        "\033[38;5;220m",  // Yellow
        "\033[38;5;226m",  // Bright Yellow
        "\033[38;5;228m"   // Light Yellow
    };

    initialize_frame();
}

void PhoenixAnimator::initialize_frame() {
    // Phoenix art (19 lines) - aligned to match ESQL base
    phoenix_frame_ = {
        "                    ...  .:....        ",
        "         ..,l:c,'.....,;;;,..          ",
        "      ..ccoo;'...      .',ll::..       ",
        "     .,xxodc..          .,:dodl,.      ",
        "   ..clkkoo,             .;oddxl;'.    ",
        "  .,:ddxxkd,.            .cddddocc:.   ",
        "  ';ooxllkx:             .lxodoxll:.   ",
        " ..cldolclxl.  . ..     .,xooldooc:'   ",
        " .,;cllc:;cdd, .:xkl,..,ddc::looolc'  ",
        "  ';loll:,,,:xl.'do:..ld:;,;cloooc'    ",
        "   .,coooc:,,;lododxddc,,,;coool;.     ",
        "     ..;clll:;,,lc:::,,;:cllc:,.       ",
        "        ..,;::c;;:,,c,::cc,...         ",
        "         ....',,::,:cc:,'...           ",
        "         ;,',;::c;,,ccc::;'.           ",
        "       ,':;,;:cccll:ccl:;''''..        ",
        "       .,,':,;ccl:c;:::,,;..,;         ",
        "       ....',,;c:;;;:;;;;.'.'.         ",
        "         ....','.......''....          "
    };
}

std::string PhoenixAnimator::apply_gradient(const std::string& line, int frame) {
    std::string result;
    
    // Create a flowing fiery gradient effect without changing the shape
    for (size_t i = 0; i < line.length(); ++i) {
        // Use frame to create flowing effect through the colors
        int color_index = (frame + (i / 2)) % fire_colors_.size();
        result += fire_colors_[color_index] + std::string(1, line[i]);
    }

    return result + "\033[0m";
}

void PhoenixAnimator::animate_fire_effect(int duration_ms) {
    std::cout << "\033[?25l"; // Hide cursor

    int frame_delay = 100;
    int total_frames = duration_ms / frame_delay;

    // Calculate positioning for phoenix on right side with spacing
    int phoenix_start_col = terminal_width_ - 45;
    if (phoenix_start_col < 40) {
        phoenix_start_col = 40; // Minimum left margin
    }

    int start_row = 1;

    for (int frame = 0; frame < total_frames; ++frame) {
        // Redraw all lines of the phoenix with fiery gradient
        for (size_t i = 0; i < phoenix_frame_.size(); ++i) {
            if (phoenix_start_col > 0) {
                std::cout << "\033[" << (start_row + i) << ";" << phoenix_start_col << "H";
            }

            std::cout << "\033[K"; // Clear the line segment
            std::string colored_line = apply_gradient(phoenix_frame_[i], frame + i);
            std::cout << colored_line;
        }

        std::cout.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(frame_delay));
    }

    std::cout << "\033[?25h"; // Show cursor
}

void PhoenixAnimator::draw_static_phoenix() {
    // Draw phoenix at a specific position (for initial display)
    int start_row = 1;
    int phoenix_start_col = terminal_width_ - 45;
    
    for (size_t i = 0; i < phoenix_frame_.size(); ++i) {
        if (phoenix_start_col > 0) {
            std::cout << "\033[" << (start_row + i) << ";" << phoenix_start_col << "H";
        }
        
        std::string colored_line = apply_gradient(phoenix_frame_[i], i * 2);
        std::cout << colored_line << "\n";
    }
}
