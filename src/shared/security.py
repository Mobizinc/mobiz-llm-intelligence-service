import hmac
import hashlib
import time
import os
from typing import Optional
import logging

try:
    from azure.keyvault.secrets import SecretClient
    from azure.identity import DefaultAzureCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    SecretClient = None
    DefaultAzureCredential = None

logger = logging.getLogger(__name__)

class SecurityManager:
    def __init__(self):
        self.key_vault_url = os.environ.get("AZURE_KEY_VAULT_URL")
        self.credential = None
        self.secret_client = None
        
        # Initialize secret cache (1-hour TTL)
        self._secret_cache = {}
        self._cache_ttl = int(os.environ.get("SECRET_CACHE_TTL", "3600"))  # 1 hour default
        self._cache_hits = 0
        self._cache_misses = 0
        
        if self.key_vault_url and AZURE_AVAILABLE:
            try:
                self.credential = DefaultAzureCredential()
                self.secret_client = SecretClient(
                    vault_url=self.key_vault_url, 
                    credential=self.credential
                )
                logger.info("Azure Key Vault client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Key Vault client: {e}")
        elif not AZURE_AVAILABLE:
            logger.info("Azure SDK not available, falling back to environment variables only")
    
    def _is_cache_valid(self, secret_name: str) -> bool:
        """Check if cached secret is still valid"""
        if secret_name not in self._secret_cache:
            return False
        
        cached_time, _ = self._secret_cache[secret_name]
        return (time.time() - cached_time) < self._cache_ttl
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret from Azure Key Vault with 1-hour caching and detailed timing"""
        start_time = time.time()
        
        # Periodically clean up expired cache (every 10th request)
        if len(self._secret_cache) > 0 and len(self._secret_cache) % 10 == 0:
            self._cleanup_expired_cache()
        
        # Check cache first
        if self._is_cache_valid(secret_name):
            self._cache_hits += 1
            _, cached_value = self._secret_cache[secret_name]
            cache_time = time.time() - start_time
            
            # Log cache stats periodically
            total_requests = self._cache_hits + self._cache_misses
            if total_requests % 10 == 0:  # Every 10 requests
                hit_rate = (self._cache_hits / total_requests) * 100
                logger.info(f"📊 Secret cache stats: {hit_rate:.1f}% hit rate ({self._cache_hits}/{total_requests}), cache size: {len(self._secret_cache)}")
            
            logger.debug(
                f"Secret '{secret_name}' retrieved from cache",
                extra={
                    "secret_name": secret_name,
                    "cache_time": f"{cache_time:.3f}s",
                    "cache_ttl": self._cache_ttl,
                    "cache_hits": self._cache_hits,
                    "source": "cache"
                }
            )
            return cached_value
        
        self._cache_misses += 1
        
        try:
            if self.secret_client:
                kv_start = time.time()
                secret = self.secret_client.get_secret(secret_name)
                kv_time = time.time() - kv_start
                
                # Cache the secret with current timestamp
                self._secret_cache[secret_name] = (time.time(), secret.value)
                
                total_time = time.time() - start_time
                if kv_time > 2.0:
                    logger.warning(
                        f"SLOW Key Vault operation for secret '{secret_name}'",
                        extra={
                            "secret_name": secret_name,
                            "kv_time": f"{kv_time:.3f}s",
                            "total_time": f"{total_time:.3f}s",
                            "cache_misses": self._cache_misses,
                            "source": "key_vault",
                            "performance_issue": True
                        }
                    )
                else:
                    logger.info(
                        f"Secret '{secret_name}' retrieved from Key Vault",
                        extra={
                            "secret_name": secret_name,
                            "kv_time": f"{kv_time:.3f}s",
                            "total_time": f"{total_time:.3f}s",
                            "cache_misses": self._cache_misses,
                            "source": "key_vault"
                        }
                    )
                
                return secret.value
        except Exception as e:
            kv_time = time.time() - start_time
            logger.warning(f"Failed to get secret '{secret_name}' from Key Vault after {kv_time:.3f}s: {e}")
            
            # Check if this is an authentication failure
            if self.is_auth_failure_error(e):
                logger.warning(f"Authentication failure detected for '{secret_name}', invalidating cache")
                self.invalidate_cache()
        
        # Fallback to environment variable
        env_start = time.time()
        env_value = os.environ.get(secret_name.upper().replace("-", "_"))
        if env_value:
            # Cache environment variable too
            self._secret_cache[secret_name] = (time.time(), env_value)
            env_time = time.time() - env_start
            total_time = time.time() - start_time
            logger.info(f"🌱 Secret '{secret_name}' from environment in {env_time:.3f}s (total: {total_time:.3f}s)")
        else:
            total_time = time.time() - start_time
            logger.error(f"❌ Secret '{secret_name}' not found anywhere after {total_time:.3f}s")
        
        return env_value
    
    def _cleanup_expired_cache(self):
        """Remove expired entries from cache"""
        current_time = time.time()
        expired_keys = [
            key for key, (cached_time, _) in self._secret_cache.items()
            if (current_time - cached_time) >= self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._secret_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired secret cache entries")
    
    def invalidate_cache(self):
        """Invalidate the entire secret cache (useful for auth failures)"""
        cache_size = len(self._secret_cache)
        self._secret_cache.clear()
        logger.warning(f"Invalidated secret cache ({cache_size} entries cleared)")
    
    def invalidate_secret(self, secret_name: str):
        """Invalidate a specific secret from cache"""
        if secret_name in self._secret_cache:
            del self._secret_cache[secret_name]
            logger.warning(f"Invalidated secret '{secret_name}' from cache")
    
    def is_auth_failure_error(self, error: Exception) -> bool:
        """Check if an error indicates authentication failure"""
        error_str = str(error).lower()
        auth_error_indicators = [
            "authentication failed",
            "unauthorized", 
            "invalid_auth",
            "token",
            "credential",
            "forbidden",
            "access denied"
        ]
        return any(indicator in error_str for indicator in auth_error_indicators)
    
    def verify_slack_signature(self, request_body: str, timestamp: str, signature: str) -> bool:
        """Verify Slack request signature"""
        slack_signing_secret = self.get_secret("slack-signing-secret")
        if not slack_signing_secret:
            logger.error("Slack signing secret not found")
            return False
        
        if abs(time.time() - int(timestamp)) > 60 * 5:
            logger.warning("Request timestamp too old")
            return False
        
        sig_basestring = f"v0:{timestamp}:{request_body}"
        expected_signature = "v0=" + hmac.new(
            slack_signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_signature, signature)
    
    def sanitize_input(self, user_input: str) -> str:
        """Basic input sanitization"""
        if not user_input:
            return ""
        
        sanitized = user_input.strip()
        
        dangerous_patterns = [
            "rm -rf", "sudo", "chmod", "chown", "passwd", 
            "wget", "curl", "nc ", "netcat", "nmap",
            "DROP TABLE", "DELETE FROM", "INSERT INTO",
            "eval(", "exec(", "system(", "subprocess",
            "<script", "javascript:", "data:text/html"
        ]
        
        for pattern in dangerous_patterns:
            if pattern.lower() in sanitized.lower():
                logger.warning(f"Potentially dangerous input detected: {pattern}")
                sanitized = sanitized.replace(pattern, "[FILTERED]")
        
        return sanitized[:2000]


# Singleton instance
_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get singleton SecurityManager instance"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager