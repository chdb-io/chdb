#pragma once

#include "chdb.h"
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <stdexcept>
#include <span>

namespace CHDB
{

extern chdb_connection * connect_chdb_with_exception(int argc, char ** argv);

/**
 * These codes provide detailed error classification for better error handling
 * and debugging. Each error code corresponds to a specific failure scenario.
 */
enum class ChdbErrorCode : std::uint8_t
{
    Success = 0, ///< Operation completed successfully
    InvalidResultHandle, ///< Invalid or null result handle provided
    ConnectionFailed, ///< Failed to establish database connection
    ConnectionClosed, ///< Operation attempted on closed connection
    QueryExecutionFailed, ///< SQL query execution failed
    StreamingQueryFailed, ///< Streaming query initialization failed
    StreamFetchFailed, ///< Failed to fetch streaming data
    CommandLineQueryFailed, ///< Command line query execution failed
    UnknownError ///< Unspecified error occurred
};

/**
 * This exception class extends std::runtime_error to provide additional
 * context through error codes. It maintains both human-readable error
 * messages and machine-readable error codes for programmatic handling.
 */
class ChdbError : public std::runtime_error {
public:
    /**
     * Constructs ChdbError with specific error code and message
     * @param code The specific error code identifying the failure type
     * @param message Human-readable error description
     */
    explicit ChdbError(ChdbErrorCode code, const char * message)
        : std::runtime_error(message)
        , error_code_(code)
    {
    }

    /**
     * Constructs ChdbError with message only (UnknownError code)
     * @param message Human-readable error description
     */
    explicit ChdbError(const char * message)
        : std::runtime_error(message)
        , error_code_(ChdbErrorCode::UnknownError)
    {
    }

    /**
     * Gets the specific error code
     * @return The error code associated with this exception
     */
    ChdbErrorCode code() const noexcept { return error_code_; }
    
private:
    ChdbErrorCode error_code_;
};

/**
 * RAII wrapper for ChDB query results
 * 
 * The Result class provides safe and convenient access to query results
 * with automatic resource management. It wraps the C API result handle
 * and ensures proper cleanup when the object is destroyed.
 */
class Result {
public:
    /**
     * Constructs Result from C API result handle
     * @param result Raw result handle from ChDB C API
     * @throws ChdbError if result handle is null
     */
    explicit Result(chdb_result* result) : result_(result) {
        if (!result_) {
            throw ChdbError(ChdbErrorCode::InvalidResultHandle, "Invalid result handle");
        }
    }

    /**
     * Destructor - automatically cleans up result resources
     */
    ~Result() {
        if (result_) {
            chdb_destroy_query_result(result_);
        }
    }

    /// Copy constructor is deleted (move-only semantics)
    Result(const Result&) = delete;
    /// Copy assignment is deleted (move-only semantics)
    Result& operator=(const Result&) = delete;

    /**
     * Move constructor - transfers ownership of result handle
     * @param other Source Result object (will be left in valid but empty state)
     */
    Result(Result&& other) noexcept : result_(other.result_) {
        other.result_ = nullptr;
    }

    /**
     * Move assignment operator - transfers ownership of result handle
     * @param other Source Result object (will be left in valid but empty state)
     * @return Reference to this object
     */
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

    /**
     * Gets result data as string_view (zero-copy)
     * @return String view of the result data, empty if result is invalid
     * @note The returned view is valid only while this Result object exists
     */
    std::string_view data() const {
        if (!result_) return {};
        char* buffer = chdb_result_buffer(result_);
        size_t length = chdb_result_length(result_);
        return std::string_view(buffer, length);
    }

    /**
     * Gets result data as byte span (zero-copy)
     * @return Span of bytes representing the result data
     * @note The returned span is valid only while this Result object exists
     */
    std::span<const std::byte> bytes() const {
        if (!result_) return {};
        char* buffer = chdb_result_buffer(result_);
        size_t length = chdb_result_length(result_);
        return std::span<const std::byte>(reinterpret_cast<const std::byte*>(buffer), length);
    }

    /**
     * Gets result data as string (copies data)
     * @return String copy of the result data
     * @note This method creates a copy, use data() for zero-copy access
     */
    std::string str() const {
        auto view = data();
        return std::string(view);
    }

    /**
     * Gets the size of result data in bytes
     * @return Size of result data, or 0 if result is invalid
     */
    size_t size() const {
        return result_ ? chdb_result_length(result_) : 0;
    }

    /**
     * Gets query execution time
     * @return Elapsed time for query execution, or 0.0 if result is invalid
     */
    double elapsed() const {
        return result_ ? chdb_result_elapsed(result_) : 0.0;
    }

    /**
     * Gets number of rows processed by the query
     * @return Number of rows read/processed, or 0 if result is invalid
     */
    uint64_t rows_read() const {
        return result_ ? chdb_result_rows_read(result_) : 0;
    }

    /**
     * Gets number of bytes processed by the query
     * @return Number of bytes read/processed, or 0 if result is invalid
     */
    uint64_t bytes_read() const {
        return result_ ? chdb_result_bytes_read(result_) : 0;
    }

    /**
     * Gets number of rows read from storage
     * @return Number of rows read from underlying storage, or 0 if result is invalid
     */
    uint64_t storage_rows_read() const {
        return result_ ? chdb_result_storage_rows_read(result_) : 0;
    }

    /**
     * Gets number of bytes read from storage
     * @return Number of bytes read from underlying storage, or 0 if result is invalid
     */
    uint64_t storage_bytes_read() const {
        return result_ ? chdb_result_storage_bytes_read(result_) : 0;
    }

    /**
     * Gets error message if query failed
     * @return Optional containing error message, or nullopt if no error occurred
     */
    std::optional<std::string> error() const {
        if (!result_) return std::nullopt;
        const char* error_msg = chdb_result_error(result_);
        return error_msg ? std::optional<std::string>(error_msg) : std::nullopt;
    }

    /**
     * Checks if result contains an error
     * @return true if an error occurred, false otherwise
     */
    bool has_error() const {
        return error().has_value();
    }

    /**
     * Throws ChdbError if result contains an error
     * @throws ChdbError if the result indicates an error occurred
     * @note This is useful for converting error results to exceptions
     */
    void throw_if_error() const {
        auto err = error();
        if (err) {
            throw ChdbError(err->c_str());
        }
    }

    /**
     * Gets the raw C API result handle
     * @return Raw chdb_result pointer (for internal use)
     * @warning This is an internal method - use with caution
     */
    chdb_result * get() const { return result_; }

private:
    chdb_result * result_; ///< Raw C API result handle
};

/**
 * The Connection class provides a high-level interface to ChDB database
 * operations. It manages the connection lifecycle automatically and provides
 * both regular and streaming query capabilities.
 */
class Connection {
public:
    /**
     * Constructs connection with custom arguments
     */
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
        chdb_connection * conn_ptr = connect_chdb_with_exception(static_cast<int>(argv.size()), argv.data());
        if (!conn_ptr)
        {
            throw ChdbError(ChdbErrorCode::ConnectionFailed, "Failed to create database connection");
        }
        conn_ = *conn_ptr;
    }

    /**
     * Constructs connection to file-based database
     */
    explicit Connection(const std::string& path) : Connection(std::vector<std::string>{"--path=" + path}) {}

    ~Connection() {
        if (conn_) {
            chdb_close_conn(&conn_);
        }
    }

    /// Copy constructor is deleted (move-only semantics)
    Connection(const Connection&) = delete;
    /// Copy assignment is deleted (move-only semantics)
    Connection& operator=(const Connection&) = delete;

    /** Move constructor - transfers connection ownership */
    Connection(Connection&& other) noexcept : conn_(other.conn_) {
        other.conn_ = nullptr;
    }

    /** Move assignment - transfers connection ownership */
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

    /** Execute SQL query and return complete result */
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

    /** Initialize streaming query for large datasets */
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

    /** Fetch next batch from streaming query */
    Result stream_fetch(Result& stream_result) const {
        if (!conn_) {
            throw ChdbError(ChdbErrorCode::ConnectionClosed, "Connection is closed");
        }

        chdb_result * result = chdb_stream_fetch_result(conn_, stream_result.get());
        if (!result) {
            throw ChdbError(ChdbErrorCode::StreamFetchFailed, "Stream fetch failed");
        }
        
        return Result(result);
    }

    /** Cancel ongoing streaming query */
    void stream_cancel(Result& stream_result) const {
        if (!conn_) {
            throw ChdbError(ChdbErrorCode::ConnectionClosed, "Connection is closed");
        }

        chdb_stream_cancel_query(conn_, stream_result.get());
    }

private:
    chdb_connection conn_;
};

/** Iterator for streaming query results */
class StreamIterator {
public:
    /** Construct iterator for active streaming query */
    StreamIterator(const Connection & conn, Result & stream_result)
        : conn_(&conn)
        , stream_result_(&stream_result)
        , finished_(false)
    {
        try
        {
            advance();
        }
        catch (...)
        {
            finished_ = true;
            throw;
        }
    }

    /** Construct end iterator */
    StreamIterator()
        : conn_(nullptr)
        , stream_result_(nullptr)
        , finished_(true)
    {
    }

    StreamIterator(StreamIterator && other) noexcept
        : conn_(other.conn_)
        , stream_result_(other.stream_result_)
        , current_result_(std::move(other.current_result_))
        , finished_(other.finished_)
    {
        other.conn_ = nullptr;
        other.stream_result_ = nullptr;
        other.finished_ = true;
    }

    StreamIterator & operator=(StreamIterator && other) noexcept
    {
        if (this != &other)
        {
            conn_ = other.conn_;
            stream_result_ = other.stream_result_;
            current_result_ = std::move(other.current_result_);
            finished_ = other.finished_;
            other.conn_ = nullptr;
            other.stream_result_ = nullptr;
            other.finished_ = true;
        }
        return *this;
    }

    // No need for custom destructor - std::optional handles cleanup automatically
    ~StreamIterator() = default;

    StreamIterator(const StreamIterator&) = delete;
    StreamIterator& operator=(const StreamIterator&) = delete;

    Result & operator*()
    {
        if (!current_result_)
        {
            throw ChdbError("Dereferencing invalid iterator");
        }
        return current_result_.value();
    }

    const Result & operator*() const
    {
        if (!current_result_)
        {
            throw ChdbError("Dereferencing invalid iterator");
        }
        return current_result_.value();
    }

    Result * operator->()
    {
        if (!current_result_)
        {
            throw ChdbError("Dereferencing invalid iterator");
        }
        return &current_result_.value();
    }

    const Result * operator->() const
    {
        if (!current_result_)
        {
            throw ChdbError("Dereferencing invalid iterator");
        }
        return &current_result_.value();
    }

    StreamIterator& operator++() {
        advance();
        return *this;
    }
    bool operator==(const StreamIterator & other) const
    {
        return finished_ == other.finished_ && (finished_ || (conn_ == other.conn_ && stream_result_ == other.stream_result_));
    }

    bool operator!=(const StreamIterator& other) const {
        return !(*this == other);
    }

private:
    void advance() {
        if (finished_ || !conn_ || !stream_result_)
            return;

        current_result_ = conn_->stream_fetch(*stream_result_);
        if (!current_result_ || current_result_->rows_read() == 0 || current_result_->has_error())
        {
            finished_ = true;
        }
    }

    const Connection* conn_;
    Result * stream_result_;
    std::optional<Result> current_result_;
    bool finished_;
};

/** Range-based for loop support for streaming queries */
class Stream {
public:
    /** Construct stream wrapper for range-based iteration */
    Stream(const Connection & conn, Result & stream_result)
        : conn_(conn)
        , stream_result_(stream_result)
    {
    }

    /** Get iterator to start of stream */
    StreamIterator begin() { return StreamIterator(conn_, stream_result_); }

    /** Get iterator representing end of stream */
    StreamIterator end() {
        return StreamIterator();
    }

private:
    const Connection& conn_;
    Result & stream_result_;
};

/** Execute query using command-line style arguments */
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

/** Execute query using C-style command-line arguments */
inline Result query_cmdline(int argc, char** argv) {
    chdb_result* result = chdb_query_cmdline(argc, argv);
    if (!result) {
        throw ChdbError(ChdbErrorCode::CommandLineQueryFailed, "Command line query execution failed");
    }
    
    return Result(result);
}

/** Convenience function to create database connection */
inline Connection connect(const std::string& path = ":memory:") {
    return Connection(path);
}

/** Convenience function to create connection with custom arguments */
inline Connection connect(const std::vector<std::string>& args) {
    return Connection(args);
}

} // namespace CHDB
