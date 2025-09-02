#pragma once

#include "chdb.h"
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <stdexcept>
#include <span>

namespace chdb {

enum class ChdbErrorCode : std::uint8_t {
    Success = 0,
    InvalidResultHandle,
    ConnectionFailed,
    ConnectionClosed,
    QueryExecutionFailed,
    StreamingQueryFailed,
    StreamFetchFailed,
    CommandLineQueryFailed,
    UnknownError
};

class ChdbError : public std::runtime_error {
public:
    explicit ChdbError(ChdbErrorCode code, const char * message)
        : std::runtime_error(message)
        , error_code_(code)
    {
    }

    explicit ChdbError(const char * message)
        : std::runtime_error(message)
        , error_code_(ChdbErrorCode::UnknownError)
    {
    }

    ChdbErrorCode code() const noexcept { return error_code_; }
    
private:
    ChdbErrorCode error_code_;
};

class Result {
public:
    explicit Result(chdb_result* result) : result_(result) {
        if (!result_) {
            throw ChdbError(ChdbErrorCode::InvalidResultHandle, "Invalid result handle");
        }
    }

    ~Result() {
        if (result_) {
            chdb_destroy_query_result(result_);
        }
    }

    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;

    Result(Result&& other) noexcept : result_(other.result_) {
        other.result_ = nullptr;
    }

    Result& operator=(Result&& other) noexcept {
        if (this != &other) {
            if (result_) {
                chdb_destroy_query_result(result_);
            }
            result_ = other.result_;
            other.result_ = nullptr;
        }
        return *this;
    }

    std::string_view data() const {
        if (!result_) return {};
        char* buffer = chdb_result_buffer(result_);
        size_t length = chdb_result_length(result_);
        return std::string_view(buffer, length);
    }
    
    std::span<const std::byte> bytes() const {
        if (!result_) return {};
        char* buffer = chdb_result_buffer(result_);
        size_t length = chdb_result_length(result_);
        return std::span<const std::byte>(reinterpret_cast<const std::byte*>(buffer), length);
    }

    std::string str() const {
        auto view = data();
        return std::string(view);
    }

    size_t size() const {
        return result_ ? chdb_result_length(result_) : 0;
    }

    double elapsed() const {
        return result_ ? chdb_result_elapsed(result_) : 0.0;
    }

    uint64_t rows_read() const {
        return result_ ? chdb_result_rows_read(result_) : 0;
    }

    uint64_t bytes_read() const {
        return result_ ? chdb_result_bytes_read(result_) : 0;
    }

    uint64_t storage_rows_read() const {
        return result_ ? chdb_result_storage_rows_read(result_) : 0;
    }

    uint64_t storage_bytes_read() const {
        return result_ ? chdb_result_storage_bytes_read(result_) : 0;
    }

    std::optional<std::string> error() const {
        if (!result_) return std::nullopt;
        const char* error_msg = chdb_result_error(result_);
        return error_msg ? std::optional<std::string>(error_msg) : std::nullopt;
    }

    bool has_error() const {
        return error().has_value();
    }

    void throw_if_error() const {
        auto err = error();
        if (err) {
            throw ChdbError(err->c_str());
        }
    }

    chdb_result* release() {
        chdb_result* tmp = result_;
        result_ = nullptr;
        return tmp;
    }

private:
    chdb_result* result_;
};

class Connection {
public:
    explicit Connection(const std::vector<std::string> & args = {})
    {
        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        static std::string chdb_program_name = "chdb";
        argv.push_back(chdb_program_name.data());
        for (const auto & arg : args)
        {
            argv.push_back(const_cast<char *>(arg.data()));
        }

        chdb_connection* conn_ptr = chdb_connect(static_cast<int>(argv.size()), argv.data());
        if (!conn_ptr)
        {
            throw ChdbError(ChdbErrorCode::ConnectionFailed, "Failed to create database connection");
        }
        conn_ = *conn_ptr;
    }

    explicit Connection(const std::string& path) : Connection(std::vector<std::string>{"--path=" + path}) {}

    ~Connection() {
        if (conn_) {
            chdb_close_conn(&conn_);
        }
    }

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    Connection(Connection&& other) noexcept : conn_(other.conn_) {
        other.conn_ = nullptr;
    }

    Connection& operator=(Connection&& other) noexcept {
        if (this != &other) {
            if (conn_) {
                chdb_close_conn(&conn_);
            }
            conn_ = other.conn_;
            other.conn_ = nullptr;
        }
        return *this;
    }

    Result query(const std::string & sql, const std::string & format = "TabSeparated") const
    {
        if (!conn_) {
            throw ChdbError(ChdbErrorCode::ConnectionClosed, "Connection is closed");
        }
        chdb_result * result = chdb_query(conn_, sql.c_str(), format.c_str());
        if (!result) {
            throw ChdbError(ChdbErrorCode::QueryExecutionFailed, "Query execution failed");
        }

        return Result(result);
    }

    Result stream_query(const std::string & sql, const std::string & format = "TabSeparated") const
    {
        if (!conn_) {
            throw ChdbError(ChdbErrorCode::ConnectionClosed, "Connection is closed");
        }

        chdb_result * result = chdb_stream_query(conn_, sql.c_str(), format.c_str());
        if (!result) {
            throw ChdbError(ChdbErrorCode::StreamingQueryFailed, "Streaming query initialization failed");
        }
        
        return Result(result);
    }

    Result stream_fetch(Result& stream_result) const {
        if (!conn_) {
            throw ChdbError(ChdbErrorCode::ConnectionClosed, "Connection is closed");
        }
        
        chdb_result* result = chdb_stream_fetch_result(conn_, stream_result.release());
        if (!result) {
            throw ChdbError(ChdbErrorCode::StreamFetchFailed, "Stream fetch failed");
        }
        
        return Result(result);
    }

    void stream_cancel(Result& stream_result) const {
        if (!conn_) {
            throw ChdbError(ChdbErrorCode::ConnectionClosed, "Connection is closed");
        }
        
        chdb_stream_cancel_query(conn_, stream_result.release());
    }

    bool is_valid() const {
        return conn_ != nullptr;
    }

private:
    chdb_connection conn_;
};

class StreamIterator {
public:
    StreamIterator(const Connection& conn, Result&& stream_result) 
        : conn_(&conn), stream_result_(std::move(stream_result)), finished_(false) {
        advance();
    }

    StreamIterator() : conn_(nullptr), stream_result_(nullptr), finished_(true) {}

    StreamIterator(StreamIterator && other) noexcept
        : conn_(other.conn_)
        , stream_result_(std::move(other.stream_result_))
        , current_result_(std::move(other.current_result_))
        , finished_(other.finished_)
    {
        other.conn_ = nullptr;
        other.finished_ = true;
    }

    StreamIterator & operator=(StreamIterator && other) noexcept
    {
        if (this != &other)
        {
            cleanup();
            conn_ = other.conn_;
            stream_result_ = std::move(other.stream_result_);
            current_result_ = std::move(other.current_result_);
            finished_ = other.finished_;
            other.conn_ = nullptr;
            other.finished_ = true;
        }
        return *this;
    }

    ~StreamIterator() { cleanup(); }

    StreamIterator(const StreamIterator&) = delete;
    StreamIterator& operator=(const StreamIterator&) = delete;

    Result& operator*() {
        return current_result_;
    }

    const Result& operator*() const {
        return current_result_;
    }

    Result* operator->() {
        return &current_result_;
    }

    const Result* operator->() const {
        return &current_result_;
    }

    StreamIterator& operator++() {
        advance();
        return *this;
    }
    bool operator==(const StreamIterator & other) const { return finished_ == other.finished_ && (finished_ || conn_ == other.conn_); }

    bool operator!=(const StreamIterator& other) const {
        return !(*this == other);
    }

private:
    void cleanup() noexcept
    {
        if (conn_ && !finished_)
        {
            try
            {
                conn_->stream_cancel(stream_result_);
            }
            catch (...)
            {
            }
        }
    }

    void advance() {
        if (finished_ || !conn_) return;
        
        try {
            current_result_ = conn_->stream_fetch(stream_result_);
            if (current_result_.size() == 0 || current_result_.has_error()) {
                finished_ = true;
            }
        } catch (const ChdbError&) {
            finished_ = true;
        }
    }

    const Connection* conn_;
    Result stream_result_;
    Result current_result_{nullptr};
    bool finished_;
};

class Stream {
public:
    Stream(const Connection& conn, Result&& stream_result) 
        : conn_(conn), stream_result_(std::move(stream_result)) {}

    StreamIterator begin() {
        return StreamIterator(conn_, std::move(stream_result_));
    }

    StreamIterator end() {
        return StreamIterator();
    }

private:
    const Connection& conn_;
    Result stream_result_;
};

inline Result query_cmdline(const std::vector<std::string>& args) {
    std::vector<char*> argv;
    std::vector<std::string> arg_storage;
    
    arg_storage.reserve(args.size());
    argv.reserve(args.size());
    
    for (const auto& arg : args) {
        arg_storage.push_back(arg);
        argv.push_back(arg_storage.back().data());
    }
    
    chdb_result* result = chdb_query_cmdline(static_cast<int>(argv.size()), argv.data());
    if (!result) {
        throw ChdbError(ChdbErrorCode::CommandLineQueryFailed, "Command line query execution failed");
    }
    
    return Result(result);
}

inline Result query_cmdline(int argc, char** argv) {
    chdb_result* result = chdb_query_cmdline(argc, argv);
    if (!result) {
        throw ChdbError(ChdbErrorCode::CommandLineQueryFailed, "Command line query execution failed");
    }
    
    return Result(result);
}

inline Connection connect(const std::string& path = ":memory:") {
    return Connection(path);
}

inline Connection connect(const std::vector<std::string>& args) {
    return Connection(args);
}

}